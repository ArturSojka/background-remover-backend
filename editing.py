import numpy as np
from PIL import Image, ImageFilter, ImageColor
import cv2
from foreground_estimation import estimate_foreground

def edit(img: Image.Image, alpha: Image.Image, settings: dict, bg = None) -> Image.Image:
    """
    Edit the image based on the options
    """
    
    if settings['blendingMethod'] == 'advanced':
        img_arr = np.array(img,dtype=np.float32)/255.0
        alpha_arr = np.array(alpha,dtype=np.float32)/255.0
        img, _ = estimate_foreground(img_arr, alpha_arr)
        img = Image.fromarray((img*255.0).astype(np.uint8))
    
    img = img.convert("RGBA")
    orig_width, orig_height = img.size

    # Create foreground with transparency
    img_np = np.array(img)
    img_np[:, :, 3] = np.array(alpha)
    foreground = Image.fromarray(img_np)

    bg_type = settings["background"]["type"]
    blur = settings["background"].get("blur", False)
    bg_color = settings["background"].get("color", "#ffffff")

    # TODO: Maybe move to separate function
    if bg_type == "none":
        background = Image.new("RGBA", (orig_width, orig_height), (0, 0, 0, 0))
    elif bg_type == "color":
        background = Image.new("RGBA", (orig_width, orig_height), color=bg_color)
    elif bg_type == "original":
        background = img.copy()
        if blur:
            background = background.filter(ImageFilter.GaussianBlur(radius=10))
        background = background.convert("RGBA")
    elif bg_type == "image" and bg:
        background = bg.convert("RGBA")
        bg_width, bg_height = background.size

        # Resize foreground to fit background
        aspect_ratio = orig_width / orig_height
        if bg_width / bg_height > aspect_ratio:
            new_height = bg_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = bg_width
            new_height = int(new_width / aspect_ratio)

        resized_foreground = foreground.resize((new_width, new_height), Image.LANCZOS)
        resized_alpha = alpha.resize((new_width, new_height), Image.LANCZOS)

        background = background.copy()
        if blur:
            background = background.filter(ImageFilter.GaussianBlur(radius=10))
        background = add_effects(background, resized_alpha, settings["effect"], offset=((bg_width - new_width)//2, (bg_height - new_height)//2))

        background.paste(resized_foreground, ((bg_width - new_width) // 2, (bg_height - new_height) // 2), mask=resized_foreground)
        return background
    else:
        raise ValueError(f"Unsupported background type or missing image: {bg_type}")

    background = add_effects(background, alpha, settings["effect"])

    # TODO: Add foreground_estimation
    result = Image.alpha_composite(background, foreground)
    return result

def add_effects(background, alpha, effect, offset=(0, 0)) -> Image.Image:
    """
    Add effects that are behind the foreground and in front of the background.
    """
    if effect["type"] == "none":
        return background

    color = effect.get("color", "#000000")
    color_rgb = ImageColor.getrgb(color)

    # TODO: Maybe add option for thickness
    mask = (np.array(alpha) > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilated = cv2.dilate(mask, kernel)

    w,h = alpha.size
    color_rgb = ImageColor.getrgb(effect.get("color", "#000000"))
    effect_img = Image.new("RGBA", (w, h), color_rgb + (255,))
    effect_mask = Image.fromarray(dilated).convert("L")
    
    # TODO: Add effect["type"] == "shadow"

    background.paste(effect_img, offset, effect_mask)
    return background