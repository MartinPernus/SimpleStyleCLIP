import os
from torchvision.utils import save_image
import copy
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
import numpy as np
import torch

import os

import clip
import numpy as np
import torch

from .e4e.psp import pSp
from .manipulator import Manipulator
from .wrapper import GeneratorWrapper

from .utils import read_image, get_delta_t, create_dir


# CLIP

E4E_PATH = 'clipstyle/pretrained/e4e_ffhq_encode.pt'

thisdir = os.path.dirname(__file__)

class Model(nn.Module):
    def __init__(self, beta_threshold=0.1, alpha=3):
        super().__init__()
        thisdir = os.path.dirname(__file__)
        G = GeneratorWrapper()
        self.clip_model, _ = clip.load("ViT-B/32", device='cpu', jit=False)
        self.manipulator = Manipulator(G, 'cpu', alpha=alpha)
        self.e4e = self.initialize_e4e()
        self.prompts = np.load(os.path.join(thisdir, 'pretrained/imagenet_templates.npy')).tolist()
        self.fs3 = np.load(os.path.join(thisdir, 'tensor/fs3_old.npy'))

        self.beta_threshold = beta_threshold
        self.neutral_text = 'A face'

    def to(self, device):
        super().to(device)
        self.manipulator.latent.to(device)
        for tensor_dict in (self.manipulator.S, self.manipulator.S_mean, self.manipulator.S_std, self.manipulator.styles):
            for key in tensor_dict:
                tensor_dict[key] = tensor_dict[key].to(device)
        return self

    def forward(self, img, target_text: str):
        img = img * 2 - 1
        latent = self.e4e(img)
        styles = self.manipulator.G.mapping_stylespace(latent)
        delta_t = get_delta_t([self.neutral_text, target_text], self.clip_model, self.prompts)
        delta_s, _ = self.get_delta_s(delta_t)
        styles = self.manipulator.manipulate(styles, delta_s)

        img_edit = self.manipulator.synthesis_from_styles(styles, 0, self.manipulator.num_images)
        img_edit = img_edit / 2 + 0.5

        img_edit = F.interpolate(img_edit, img.size(-1), mode='area')
        return img_edit

    def initialize_e4e(self):
        ckpt = torch.load(E4E_PATH, map_location='cpu')
        opts = ckpt['opts']
        opts = Namespace(**opts)
        opts.batch_size = 1
        opts.ckpt = E4E_PATH

        e4e = pSp(opts)
        e4e.eval()
        e4e = e4e.to(opts.device)

        e4e.requires_grad_(False)
        return e4e

    @property
    def device(self):
        return next(self.parameters())

    def get_delta_s(self, delta_t):
        delta_s = np.dot(self.fs3, delta_t)

        select = np.abs(delta_s) < self.beta_threshold # apply beta threshold (disentangle)
        delta_s[select] = 0 # threshold 미만의 style direction을 0으로 
        num_channel = np.sum(~select)

        absmax = np.abs(copy.deepcopy(delta_s)).max()
        delta_s /= absmax # normalize

        # delta_s -> style dict
        dic = dict()
        ind = 0
        for layer in self.manipulator.G.style_layers: # 26
            dim = self.manipulator.styles[layer].shape[-1]
            if layer in self.manipulator.manipulate_layers:
                dic[layer] = torch.from_numpy(delta_s[ind:ind+dim]).to(self.device)
                ind += dim
            else:
                dic[layer] = torch.zeros([dim]).to(self.device)
        return dic, num_channel



if __name__ == '__main__':
    device = 'cuda:0'
    dir_samples = 'samples'
    imgs = os.listdir(dir_samples)

    for beta_threshold in [0.05]:
        for alpha in [1]:
            model = Model(beta_threshold=beta_threshold, alpha=alpha).to(device)
            outdir = create_dir('results', f'alpha={alpha}-beta={beta_threshold:.2f}')
    
            for i, img in enumerate(imgs):
                img = read_image(os.path.join(dir_samples, img)).to(device)
                img_edit = model(img, 'old face')
                save_image(img_edit.cpu(), os.path.join(outdir, f'{i:02d}.jpg'))