import torch
from torchvision.utils import save_image

import argparse
from clipstyle.model import Model

from clipstyle.utils import read_image


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--text', type=str, default=None)
parser.add_argument('--img', type=str, default=None)
args = parser.parse_args()

text = args.text or 'a_face_with_curly_hair'
text = ' '.join(text.split('_'))
img = args.img or 'input/1.jpg'

device = f'cuda:{args.gpu}'
model = Model().to(device)

img = read_image(img).to(device)
img_edit = model(img, text)

grid = torch.cat((img, img_edit))
save_image(grid, f'output.jpg')
print('Done.')