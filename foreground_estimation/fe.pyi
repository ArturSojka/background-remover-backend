import numpy as np
from typing import Tuple

def estimate_foreground(
        I: np.ndarray,
        alpha: np.ndarray,
        low_res_iter: int = 10,
        hi_res_iter: int = 2,
        low_size: int = 32,
        omega: float = 0.1,
        epsilon: float = 5e-3
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast multi-level foreground estimation.
    
    See `https://arxiv.org/pdf/2006.14970` for reference.
    
    Parameters
    ----------
    I : np.ndarray
        Input image normalized to `[0,1]` of shape `(H, W, 3)`.
    alpha : np.ndarray
        Alpha matte of the image normalized to `[0,1]` of shape `(H, W)`.
    low_res_iter : int, default: 10
        How many iterations to perform at low resolution.
    hi_res_iter : int, default: 2
        How many iterations to perform at high resolution.
    low_size : int, default: 32
        Below what size the resolution is considered "low".
    omega : float, default: 0.1
        Constant to control the influence of the alpha gradient.
    epsilon : float, default: 5e-3
        Regularization factor.

    Returns
    -------
    F : np.ndarray
        Estiamted foreground.
    B : np.ndarray
        Estimated background.

    Examples
    --------
    >>> F, B = estimate_foreground(I,alpha)
    """
    ...