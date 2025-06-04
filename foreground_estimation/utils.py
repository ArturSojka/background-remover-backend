import numpy as np

def blend(foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Blend the foreground with the background using the alpha matte.
    
    All inputs must have the same first two dimentions and `alpha` must be normalized to `[0,1]`.
    """
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]

    return alpha * foreground + (1 - alpha) * background

def apply_alpha(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Combine an image with an alpha channel.
    
    Both inputs must have the same first two dimentions and have the same `dtype`.
    """
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]
        
    return np.concatenate([image, alpha], axis=2)