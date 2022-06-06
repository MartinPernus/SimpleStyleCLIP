# Minimalistic StyleCLIP
This repository contains the minimal code to do text-based face editing based on the [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) method. 

## Requirements
CUDA supported GPU, Python3.7+

## Instructions
```
pip install -r requirements.txt
```
## Example Usage
```
python main.py --gpu 0 --text a_face_with_curly_hair --img input/1.jpg
```

![Example](output.jpg)