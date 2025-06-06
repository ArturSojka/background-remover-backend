import h5py
from scipy.ndimage import grey_dilation, grey_erosion
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision.transforms import Compose, ToTensor, Normalize

class SelfSupervisedDataset(Dataset):
    def __init__(self, file_path:str):
        super().__init__()
        self.file_path = file_path
        self.file = None
        self.im_size = 512
        self.transform_image = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        with h5py.File(file_path, 'r') as h5f:
            self.length = len(h5f['images'])
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        image = self.file['images'][idx]
        image = self.transform_image(image)
        return image

class NaturalDataset(Dataset):
    def __init__(self, file_path:str, split:str):
        super().__init__()
        self.file_path = file_path
        self.split = split
        self.file = None
        self.im_size = 512
        self.transform_image = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform_mask = ToTensor()
        with h5py.File(file_path, 'r') as h5f:
            self.length = len(h5f[f'{split}_images'])
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        image = self.file[f'{self.split}_images'][idx]
        mask = self.file[f'{self.split}_masks'][idx]

        trimap = (mask >= 0.9*255.0).astype(np.float32)
        not_bg = (mask > 0).astype(np.float32)

        d_size = self.im_size // 256 * random.randint(10, 20)
        e_size = self.im_size // 256 * random.randint(10, 20)

        trimap[np.where((grey_dilation(not_bg, size=(d_size, d_size)) - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
        
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        trimap = self.transform_mask(trimap)

        return image, mask, trimap
    
class SyntheticDataset(Dataset):
    def __init__(self, fg_file_path:str, split:str, bg_file_path:str, bg_per_fg:int):
        super().__init__()
        self.fg_file_path = fg_file_path
        self.bg_file_path = bg_file_path
        self.split = split
        self.bg_per_fg = bg_per_fg
        self.fg_file = None
        self.bg_file = None
        self.im_size = 512
        self.transform_image = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform_mask = ToTensor()
        with h5py.File(fg_file_path, 'r') as h5f:
            self.fg_length = len(h5f[f'{split}_images'])
            self.length = self.fg_length*bg_per_fg
        with h5py.File(bg_file_path, 'r') as h5f:
            self.bg_length = len(h5f[split])
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.fg_file is None:
            self.fg_file = h5py.File(self.fg_file_path, 'r')
        if self.bg_file is None:
            self.bg_file = h5py.File(self.bg_file_path, 'r')
        
        background = self.bg_file[self.split][random.randint(0,self.bg_length-1)]
        foreground = self.fg_file[f'{self.split}_images'][idx % self.fg_length]
        mask = self.fg_file[f'{self.split}_masks'][idx % self.fg_length]
        mask = mask.astype(np.float32)/255.0
        fmask = mask[:,:,np.newaxis]
        image = (foreground*fmask + background*(1-fmask)).astype(np.uint8)

        trimap = (mask >= 0.9).astype(np.float32)
        not_bg = (mask > 0).astype(np.float32)

        d_size = self.im_size // 256 * random.randint(10, 20)
        e_size = self.im_size // 256 * random.randint(10, 20)

        trimap[np.where((grey_dilation(not_bg, size=(d_size, d_size)) - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
        
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        trimap = self.transform_mask(trimap)

        return image, mask, trimap

def create_train_dataloader(natural_path, synthetic_path, bg_path, batch_size=8):
    natural_dataset = NaturalDataset(natural_path, "train")
    synthetic_dataset = SyntheticDataset(synthetic_path, "train", bg_path, bg_per_fg=40)
    train_loader = DataLoader(ConcatDataset([natural_dataset,synthetic_dataset]), batch_size=batch_size, shuffle=True)
    return train_loader

def create_valid_dataloader(natural_path, synthetic_path, bg_path, batch_size=8):
    natural_dataset = NaturalDataset(natural_path, "valid")
    synthetic_dataset = SyntheticDataset(synthetic_path, "valid", bg_path, bg_per_fg=10)
    valid_loader = DataLoader(ConcatDataset([natural_dataset,synthetic_dataset]), batch_size=batch_size, shuffle=False)
    return valid_loader

def create_self_supervised_dataloader(file_path, batch_size=8):
    dataset = SelfSupervisedDataset(file_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader