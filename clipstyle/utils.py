import torch
import shutil
import numpy as np
import clip
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from pathlib import Path
from typing import Union
import os

def read_image(path: Union[str, Path], unsqueeze=True):
    path = os.path.expanduser(path)
    img = to_tensor(Image.open(path).convert('RGB'))
    if unsqueeze:
        img = img.unsqueeze(0)
    return img


def zeroshot_classifier(classnames, templates, model):
    device = model.token_embedding.weight.device
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts) #tokenize
            texts = texts.to(device)
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def get_delta_t(
        classnames, 
        model, 
        prompts
    ):
    text_features = zeroshot_classifier(classnames, prompts, model).t()

    delta_t = (text_features[0] - text_features[1]).cpu().numpy()
    delta_t = delta_t/np.linalg.norm(delta_t)
    return delta_t


def create_dir(*args, remove_files=False):
    folder = Path(os.path.join(*args))
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif os.path.exists(folder) and remove_files:
        files = folder.glob('*')
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
            else:
                shutil.rmtree(file)
    else:
        pass

    return folder