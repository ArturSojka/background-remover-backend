import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from MODNet.src.models.modnet import MODNet

im_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

if torch.cuda.is_available():
    modnet = modnet.cuda()
    weights = torch.load("MODNet/pretrained/modnet_photographic_portrait_matting.ckpt")
else:
    weights = torch.load("MODNet/pretrained/modnet_photographic_portrait_matting.ckpt", map_location=torch.device('cpu'))
modnet.load_state_dict(weights)
modnet.eval()
ref_size = 512

# Code adapted from MODNet/demo/image_matting/colab/inference.py
def infer_image(im: Image.Image) -> np.ndarray:
    """
    Infer the alpha matte with MODNet.
    
    The result is a numpy array normalized to `[0,1].
    """
    
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte

# TODO: Add video inference based on MODNet/demo/video_matting/custom/run.py
# Optionally add the OFD trick as described in the MODNet paper (https://arxiv.org/pdf/2011.11961v4 page 7)