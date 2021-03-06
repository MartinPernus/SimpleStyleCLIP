# Minimalistic StyleCLIP
This repository contains the minimal code to do text-based face editing based on the [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) method. 

## Requirements
CUDA supported GPU, Python3.7+

## Instructions
### 1. Install required python libraries
```
pip install -r requirements.txt
```
### 2. Download model checkpoints
Download https://www.dropbox.com/s/sutyapujaphgaau/e4e_ffhq_encode.pt?dl=0 and put it in clipstyle/pretrained/

Download https://www.dropbox.com/s/qdljpc1ssmd0ai5/ffhq_statedict.pt?dl=0 and put it in clipstyle/pretrained


## Example Usage
```
python main.py --gpu 0 --text "a face with curly hair" --img input/1.jpg
```

![Example](output.jpg)

## Acknowledgements
The basis for this projects are the following repositories
[StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch)
[StyleCLIP-pytorch](https://github.com/soushirou/StyleCLIP-pytorch)
[E4e](https://github.com/omertov/encoder4editing)

