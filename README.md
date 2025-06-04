# Backend for background removal project

## Setup

Create a Python 3.11 virtual enviroment and activate it, then run:
```bash
pip install -r requirements.txt
python foreground_estimation/setup.py build_ext --inplace
```

You can then remove the `build` folder.

Next, download the official pre-trained MODNet from this [link](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing) (modnet_photographic_portrait_matting.ckpt) and place it in `./MODNet/pretrained`.