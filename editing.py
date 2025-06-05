import numpy as np
from PIL import Image, ImageFilter, ImageColor, ImageEnhance
import cv2

def edit(img: Image.Image, alpha: Image.Image, settings: dict, bg = None) -> Image.Image:
    """
    Edit the image based on the options
    """
    img = img.convert("RGBA")
    orig_width, orig_height = img.size

    # Create foreground with transparency
    img_np = np.array(img)
    img_np[:, :, 3] = np.array(alpha)
    foreground = Image.fromarray(img_np)

    # Create background using separate function
    background = create_background(img, settings, bg, orig_width, orig_height)
    
    # Handle special case for image background type
    if settings["background"]["type"] == "image" and bg:
        return handle_image_background(foreground, alpha, background, settings, orig_width, orig_height)

    background = add_effects(background, alpha, settings["effect"])

    # Apply foreground estimation if enabled
    if settings.get("foreground_estimation", {}).get("enabled", False):
        foreground = apply_foreground_estimation(foreground, alpha, settings["foreground_estimation"])

    result = Image.alpha_composite(background, foreground)
    return result

def create_background(img: Image.Image, settings: dict, bg: Image.Image, width: int, height: int) -> Image.Image:
    """
    Create background based on settings type
    """
    bg_type = settings["background"]["type"]
    blur = settings["background"].get("blur", False)
    bg_color = settings["background"].get("color", "#ffffff")

    if bg_type == "none":
        background = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    elif bg_type == "color":
        background = Image.new("RGBA", (width, height), color=bg_color)
    elif bg_type == "original":
        background = img.copy()
        if blur:
            background = background.filter(ImageFilter.GaussianBlur(radius=10))
        background = background.convert("RGBA")
    elif bg_type == "image" and bg:
        background = bg.convert("RGBA")
        if blur:
            background = background.filter(ImageFilter.GaussianBlur(radius=10))
    else:
        raise ValueError(f"Unsupported background type or missing image: {bg_type}")
    
    return background

def handle_image_background(foreground: Image.Image, alpha: Image.Image, background: Image.Image, settings: dict, orig_width: int, orig_height: int) -> Image.Image:
    """
    Handle special processing for image background type
    """
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

    offset = ((bg_width - new_width)//2, (bg_height - new_height)//2)
    background = add_effects(background, resized_alpha, settings["effect"], offset=offset)

    background.paste(resized_foreground, offset, mask=resized_foreground)
    return background

def add_effects(background: Image.Image, alpha: Image.Image, effect: dict, offset=(0, 0)) -> Image.Image:
    """
    Add effects that are behind the foreground and in front of the background.
    """
    if effect["type"] == "none":
        return background

    color = effect.get("color", "#000000")
    color_rgb = ImageColor.getrgb(color)
    thickness = effect.get("thickness", 10)

    # Create mask from alpha
    mask = (np.array(alpha) > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    
    w, h = alpha.size

    if effect["type"] == "outline":
        # Create outline effect
        dilated = cv2.dilate(mask, kernel)
        outline_mask = dilated - mask
        effect_mask = Image.fromarray(outline_mask).convert("L")
        effect_img = Image.new("RGBA", (w, h), color_rgb + (255,))
        
    elif effect["type"] == "glow":
        # Create glow effect
        dilated = cv2.dilate(mask, kernel)
        effect_mask = Image.fromarray(dilated).convert("L")
        # Apply gaussian blur for glow effect
        effect_mask = effect_mask.filter(ImageFilter.GaussianBlur(radius=thickness//2))
        effect_img = Image.new("RGBA", (w, h), color_rgb + (255,))
        
    else:
        # Default behavior (original dilate)
        dilated = cv2.dilate(mask, kernel)
        effect_mask = Image.fromarray(dilated).convert("L")
        effect_img = Image.new("RGBA", (w, h), color_rgb + (255,))

    background.paste(effect_img, offset, effect_mask)
    return background

def apply_foreground_estimation(foreground: Image.Image, alpha: Image.Image, estimation_settings: dict) -> Image.Image:
    """
    Apply foreground estimation to improve edge quality
    """
    method = estimation_settings.get("method", "trimap")
    
    if method == "trimap":
        # Simple trimap-based foreground estimation
        alpha_np = np.array(alpha)
        
        # Create trimap: 0 = background, 128 = unknown, 255 = foreground
        trimap = np.zeros_like(alpha_np)
        trimap[alpha_np > 200] = 255  # Definite foreground
        trimap[(alpha_np > 50) & (alpha_np <= 200)] = 128  # Unknown region
        
        # Apply simple color correction in unknown regions
        fg_np = np.array(foreground)
        unknown_mask = trimap == 128
        
        if np.any(unknown_mask):
            # Simple color spill correction
            fg_np[unknown_mask] = enhance_foreground_colors(fg_np[unknown_mask])
        
        foreground = Image.fromarray(fg_np)
    
    elif method == "color_spill_removal":
        # Remove color spill from background
        spill_color = estimation_settings.get("spill_color", [0, 255, 0])  # Default green
        tolerance = estimation_settings.get("tolerance", 30)
        
        foreground = remove_color_spill(foreground, alpha, spill_color, tolerance)
    
    return foreground

def enhance_foreground_colors(pixels: np.ndarray) -> np.ndarray:
    """
    Enhance foreground colors in uncertain regions
    """
    # Simple enhancement - increase saturation
    enhanced = pixels.copy()
    for i in range(len(enhanced)):
        if len(enhanced[i]) >= 3:  # RGB channels
            # Convert to HSV, enhance saturation, convert back
            pixel_img = Image.fromarray(enhanced[i:i+1].reshape(1, 1, -1)[:,:,:3].astype(np.uint8))
            enhancer = ImageEnhance.Color(pixel_img)
            enhanced_pixel = enhancer.enhance(1.2)  # Increase saturation by 20%
            enhanced[i][:3] = np.array(enhanced_pixel)[0, 0]
    
    return enhanced

def remove_color_spill(foreground: Image.Image, alpha: Image.Image, spill_color: list, tolerance: int) -> Image.Image:
    """
    Remove color spill from foreground
    """
    fg_np = np.array(foreground)
    alpha_np = np.array(alpha)
    
    # Create mask for semi-transparent regions where spill might occur
    spill_mask = (alpha_np > 0) & (alpha_np < 255)
    
    if np.any(spill_mask):
        # Calculate color distance from spill color
        color_diff = np.sqrt(np.sum((fg_np[spill_mask, :3] - spill_color)**2, axis=1))
        spill_pixels = color_diff < tolerance
        
        if np.any(spill_pixels):
            # Simple spill removal - desaturate spill areas
            spill_indices = np.where(spill_mask)[0][spill_pixels]
            for idx in spill_indices:
                y, x = np.unravel_index(idx, alpha_np.shape)
                # Desaturate by averaging with grayscale
                gray = np.mean(fg_np[y, x, :3])
                fg_np[y, x, :3] = (fg_np[y, x, :3] * 0.7 + gray * 0.3).astype(np.uint8)
    
    return Image.fromarray(fg_np)
