import h5py
import cv2
import os
import numpy as np
import random
from tqdm import tqdm

class BGProcessor:
    """20k background images"""
    
    def __init__(self):
        self.outfile = "D:\\backgrounds.h5"
        self.target_size = 512
        
    def crop_and_resize(self, image):
        height, width, _ = image.shape
        smaller_dim = min(width, height)

        x_offset = random.randint(0, width - smaller_dim)
        y_offset = random.randint(0, height - smaller_dim)

        image_cropped = image[y_offset:y_offset + smaller_dim, x_offset:x_offset + smaller_dim, :]

        if smaller_dim != self.target_size:
            image_cropped = cv2.resize(image_cropped, (self.target_size,self.target_size), interpolation=cv2.INTER_CUBIC)
            
        return image_cropped
    
    def process(self):
        train_folder = "D:\\BG-20k\\train"
        valid_folder = "D:\\BG-20k\\testval"
        with h5py.File(self.outfile, "w") as h5f:
            train_set = h5f.create_dataset("train", (15000, 512, 512, 3), dtype="uint8")
            valid_set = h5f.create_dataset("valid", (5000, 512, 512, 3), dtype="uint8")
            
            train_images = open('D:\\BG-20k\\train.txt','r')
            idx = 0
            for line in tqdm(train_images,desc="Train images",total=15000):
                file = os.path.join(train_folder,line.strip())
                img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
                train_set[idx] = self.crop_and_resize(img)
                idx += 1
            train_images.close()
            
            valid_images = open('D:\\BG-20k\\testval.txt','r')
            idx = 0
            for line in tqdm(valid_images,desc="Valid images",total=5000):
                file = os.path.join(valid_folder,line.strip())
                img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
                valid_set[idx] = self.crop_and_resize(img)
                idx += 1
            valid_images.close()
                
        print(f"Data saved to {self.outfile}.")
        
class SSProcessor:
    """30k images for self-supervised learning"""
    
    def __init__(self):
        self.outfile = "D:\\self-supervised.h5"
        self.target_size = 512
        
    def resize(self, image):
        height, width, _ = image.shape
        smaller_dim = min(width, height)

        if smaller_dim != self.target_size:
            image = cv2.resize(image, (self.target_size,self.target_size), interpolation=cv2.INTER_CUBIC)
            
        return image
    
    def process(self):
        animals_folder = "D:\\self-supervised\\AFHQv2"
        humans_folder = "D:\\self-supervised\\FFHQ"
        with h5py.File(self.outfile, "w") as h5f:
            dataset = h5f.create_dataset("images", (30000, 512, 512, 3), dtype="uint8")
            
            animals_pbar = tqdm(desc="Animals",total=15000)
            idx = 0
            for root, _, files in os.walk(animals_folder):
                for name in files:
                    if idx >= 15000:
                        break
                    file = os.path.join(root,name)
                    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
                    dataset[idx] = self.resize(img)
                    idx += 1
                    animals_pbar.update()
            animals_pbar.close()
                
            humans_pbar = tqdm(desc="Humans",total=15000)
            for root, _, files in os.walk(humans_folder):
                for name in files:
                    if idx >= 30000:
                        break
                    file = os.path.join(root,name)
                    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
                    dataset[idx] = self.resize(img)
                    idx += 1
                    humans_pbar.update()
            humans_pbar.close()
                
        print(f"Data saved to {self.outfile}.")
        
class NaturalProcessor:
    
    def __init__(self):
        self.outfile = "D:\\natural.h5"
        self.target_size = 512
        
    def train_pairs(self):
        """23287 images"""
        folders = [
            ("D:\\natural\\AM-2k\\train\\original", "D:\\natural\\AM-2k\\train\\mask"),
            ("D:\\natural\\IHM\\train\\image", "D:\\natural\\IHM\\train\\alpha"),
            ("D:\\natural\\PM-10k\\train\\blurred_image", "D:\\natural\\PM-10k\\train\\mask"),
        ]
        for image_folder, mask_folder in folders:
            names = [i.replace('.png','') for i in os.listdir(mask_folder)]
            for name in names:
                yield os.path.join(image_folder,name+'.jpg'), os.path.join(mask_folder,name+'.png')
        
    def valid_pairs(self):
        """1900 images"""
        folders = [
            ("D:\\natural\\AM-2k\\validation\\original", "D:\\natural\\AM-2k\\validation\\mask"),
            ("D:\\natural\\IHM\\test\\image", "D:\\natural\\IHM\\test\\alpha"),
            ("D:\\natural\\IHM\\valid\\image", "D:\\natural\\IHM\\valid\\alpha"),
            ("D:\\natural\\PM-10k\\validation\\P3M-500-NP\\original_image", "D:\\natural\\PM-10k\\validation\\P3M-500-NP\\mask"),
            ("D:\\natural\\PM-10k\\validation\\P3M-500-P\\blurred_image", "D:\\natural\\PM-10k\\validation\\P3M-500-P\\mask"),
        ]
        for image_folder, mask_folder in folders:
            names = [i.replace('.png','') for i in os.listdir(mask_folder)]
            for name in names:
                yield os.path.join(image_folder,name+'.jpg'), os.path.join(mask_folder,name+'.png')
                
    def crop_and_resize(self, image, mask):
        if image.shape[:2] != mask.shape[:2]:
            if abs(image.shape[0]/image.shape[1] - mask.shape[0]/mask.shape[1]) > 0.015:
                raise ValueError(f"Wrong shapes: {image.shape[:2]} and {mask.shape[:2]}")
            mask = cv2.resize(mask, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        image_h, image_w = image.shape[:2]        

        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or len(ys) == 0:
            print("No object found")
            return None, None

        x_min, x_max = max(0, xs.min() - 3), min(image_w - 1, xs.max() + 3)
        y_min, y_max = max(0, ys.min() - 3), min(image_h - 1, ys.max() + 3)

        pad_left = x_min
        pad_right = image_w - 1 - x_max
        pad_top = y_min
        pad_bottom = image_h - 1 - y_max
        
        padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                          borderType=cv2.BORDER_REFLECT_101)
        padded_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right,
                                         borderType=cv2.BORDER_REFLECT_101)
        
        x_min += pad_left
        x_max += pad_left
        y_min += pad_top
        y_max += pad_top
        
        padded_h, padded_w = padded_image.shape[:2]
        side = max(x_max - x_min + 1, y_max - y_min + 1)
        side = min(side,padded_h,padded_w)
        
        sq_left = max(0, (x_min+x_max)//2 - side//2)
        sq_right = sq_left + side - 1
        if sq_right >= padded_w:
            sq_left = padded_w - side
            sq_right = padded_w - 1
            
        if side > y_max-y_min:
            sq_top = max(0, (y_min+y_max)//2 - side//2)
        else:
            sq_top = y_min
        sq_bottom = sq_top + side - 1
        if sq_bottom >= padded_h:
            sq_top = padded_h - side
            sq_bottom = padded_h - 1
            
        img_x_min, img_x_max = pad_left, pad_left + image_w - 1
        img_y_min, img_y_max = pad_top, pad_top + image_h - 1
        
        if (sq_left >= img_x_min and sq_right <= img_x_max and
            sq_top >= img_y_min and sq_bottom <= img_y_max):
            new_side = min(image_w, image_h)

            sq_left = max(img_x_min, min(sq_left + side//2 - new_side//2, img_x_max - new_side))
            sq_right = sq_left + new_side - 1
            sq_top = max(img_y_min, min(sq_top + side//2 - new_side//2, img_y_max - new_side))
            sq_bottom = sq_top + new_side - 1
            side = new_side
            
        cropped_img = padded_image[sq_top:sq_bottom+1, sq_left:sq_right+1]
        assert cropped_img.shape[0] == cropped_img.shape[1]
        if cropped_img.shape[0] != self.target_size:
            cropped_img = cv2.resize(cropped_img, (self.target_size,self.target_size), interpolation=cv2.INTER_CUBIC)
            
        cropped_mask = padded_mask[sq_top:sq_bottom+1, sq_left:sq_right+1]
        assert cropped_mask.shape[0] == cropped_mask.shape[1]
        if cropped_mask.shape[0] != self.target_size:
            cropped_mask = cv2.resize(cropped_mask, (self.target_size,self.target_size), interpolation=cv2.INTER_CUBIC)
        
        return cropped_img, cropped_mask
    
    def process(self):
        with h5py.File(self.outfile, "w") as h5f:
            train_images = h5f.create_dataset("train_images", (23287, 512, 512, 3), dtype="uint8")
            train_masks = h5f.create_dataset("train_masks", (23287, 512, 512), dtype="uint8")
            valid_images = h5f.create_dataset("valid_images", (1900, 512, 512, 3), dtype="uint8")
            valid_masks = h5f.create_dataset("valid_masks", (1900, 512, 512), dtype="uint8")
            
            idx = 0
            for image_path, mask_path in tqdm(self.train_pairs(),desc="Train pairs",total=23287):
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image_cropped, mask_cropped = self.crop_and_resize(image, mask)
                train_images[idx] = image_cropped
                train_masks[idx] = mask_cropped
                idx += 1
            
            idx = 0
            for image_path, mask_path in tqdm(self.valid_pairs(),desc="Valid pairs",total=1900):
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image_cropped, mask_cropped = self.crop_and_resize(image, mask)
                valid_images[idx] = image_cropped
                valid_masks[idx] = mask_cropped
                idx += 1
                
        print(f"Data saved to {self.outfile}.")
                
class SyntheticProcessor:
    
    def __init__(self):
        self.outfile = "D:\\synthetic.h5"
        self.target_size = 512
        
    def train_pairs(self):
        """1552 images"""
        folders = [
            ("D:\\synthetic\\Distinctions-646\\Train\\FG", "D:\\synthetic\\Distinctions-646\\Train\\GT"),
            ("D:\\synthetic\\MGM\\image", "D:\\synthetic\\MGM\\alpha"),
            ("D:\\synthetic\\Transparent-460\\Train\\fg", "D:\\synthetic\\Transparent-460\\Train\\alpha"),
        ]
        folders.extend(
            (f"D:\\synthetic\\SIMD\\train\\{t}\\fg",f"D:\\synthetic\\SIMD\\train\\{t}\\alpha")
            for t in ["defocus", "flower", "fur", "glass_ice", "hair_hard", "insect", "leaf", "motion", "sharp", "silk", "water_spray"]
        )
        for image_folder, mask_folder in folders:
            masks = sorted(os.listdir(mask_folder))
            images = sorted(os.listdir(image_folder))
            for image_name, mask_name in zip(images,masks):
                yield os.path.join(image_folder,image_name), os.path.join(mask_folder,mask_name)
        
    def valid_pairs(self):
        """90 images"""
        folders = [
            ("D:\\synthetic\\Distinctions-646\\Test\\FG", "D:\\synthetic\\Distinctions-646\\Test\\GT"),
            ("D:\\synthetic\\Transparent-460\\Test\\fg", "D:\\synthetic\\Transparent-460\\Test\\alpha")
        ]
        folders.extend(
            (f"D:\\synthetic\\SIMD\\test\\{t}\\fg",f"D:\\synthetic\\SIMD\\test\\{t}\\alpha")
            for t in ["defocus", "flower", "glass_ice", "insect", "leaf", "motion", "sharp", "silk", "water_spray"]
        )
        for image_folder, mask_folder in folders:
            masks = sorted(os.listdir(mask_folder))
            images = sorted(os.listdir(image_folder))
            for image_name, mask_name in zip(images,masks):
                yield os.path.join(image_folder,image_name), os.path.join(mask_folder,mask_name)
                
    def crop_and_resize(self,image,mask):
        image_h, image_w = image.shape[:2]

        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or len(ys) == 0:
            print("No object found")
            return None, None

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        max_bbox = max(bbox_w, bbox_h)
        min_img = min(image_h, image_w)

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        if max_bbox > min_img:
            pad_size = max_bbox
            pad_left = pad_right = pad_top = pad_bottom = 0

            def edge_occupancy(edge_pixels):
                total = len(edge_pixels)
                return 100.0 * np.count_nonzero(edge_pixels) / total if total > 0 else 0.0

            left_occ = edge_occupancy(mask[:, 0])
            right_occ = edge_occupancy(mask[:, -1])
            top_occ = edge_occupancy(mask[0, :])
            bottom_occ = edge_occupancy(mask[-1, :])

            if pad_size > image_w:
                total_pad_w = pad_size - image_w
                if left_occ < 5.0 and right_occ < 5.0:
                    pad_left = total_pad_w // 2
                    pad_right = total_pad_w - pad_left
                elif left_occ < 5.0:
                    pad_left = total_pad_w
                elif right_occ < 5.0:
                    pad_right = total_pad_w
                else:
                    left_weight = max(0.01, 1.0 / left_occ)
                    right_weight = max(0.01, 1.0 / right_occ)
                    total_weight = left_weight + right_weight
                    pad_left = int(total_pad_w * (left_weight / total_weight))
                    pad_right = total_pad_w - pad_left

            if pad_size > image_h:
                total_pad_h = pad_size - image_h
                if top_occ < 5.0 and bottom_occ < 5.0:
                    pad_top = total_pad_h // 2
                    pad_bottom = total_pad_h - pad_top
                elif top_occ < 5.0:
                    pad_top = total_pad_h
                elif bottom_occ < 5.0:
                    pad_bottom = total_pad_h
                else:
                    top_weight = max(0.01, 1.0 / top_occ)
                    bottom_weight = max(0.01, 1.0 / bottom_occ)
                    total_weight = top_weight + bottom_weight
                    pad_top = int(total_pad_h * (top_weight / total_weight))
                    pad_bottom = total_pad_h - pad_top

            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=0)

            center_x += pad_left
            center_y += pad_top
            image_h, image_w = image.shape[:2]

        crop_size = min(image_h, image_w)
        half_crop = crop_size // 2

        start_x = max(0, center_x - half_crop)
        start_y = max(0, center_y - half_crop)

        end_x = min(start_x + crop_size, image_w)
        end_y = min(start_y + crop_size, image_h)
        start_x = end_x - crop_size
        start_y = end_y - crop_size

        cropped_img = image[start_y:end_y, start_x:end_x]
        assert cropped_img.shape[0] == cropped_img.shape[1]
        if cropped_img.shape[0] != self.target_size:
            cropped_img = cv2.resize(cropped_img, (self.target_size,self.target_size), interpolation=cv2.INTER_CUBIC)
            
        cropped_mask = mask[start_y:end_y, start_x:end_x]
        assert cropped_mask.shape[0] == cropped_mask.shape[1]
        if cropped_mask.shape[0] != self.target_size:
            cropped_mask = cv2.resize(cropped_mask, (self.target_size,self.target_size), interpolation=cv2.INTER_CUBIC)
        
        return cropped_img, cropped_mask
    
    def process(self):
        with h5py.File(self.outfile, "w") as h5f:
            train_images = h5f.create_dataset("train_images", (1552, 512, 512, 3), dtype="uint8")
            train_masks = h5f.create_dataset("train_masks", (1552, 512, 512), dtype="uint8")
            valid_images = h5f.create_dataset("valid_images", (90, 512, 512, 3), dtype="uint8")
            valid_masks = h5f.create_dataset("valid_masks", (90, 512, 512), dtype="uint8")
            
            idx = 0
            for image_path, mask_path in tqdm(self.train_pairs(),desc="Train pairs",total=1552):
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image_cropped, mask_cropped = self.crop_and_resize(image, mask)
                train_images[idx] = image_cropped
                train_masks[idx] = mask_cropped
                idx += 1
            
            idx = 0
            for image_path, mask_path in tqdm(self.valid_pairs(),desc="Valid pairs",total=90):
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image_cropped, mask_cropped = self.crop_and_resize(image, mask)
                valid_images[idx] = image_cropped
                valid_masks[idx] = mask_cropped
                idx += 1
                
        print(f"Data saved to {self.outfile}.")