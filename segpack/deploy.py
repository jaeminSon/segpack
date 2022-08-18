import os
from pathlib import Path
import numpy as np

import torch

from .utils import load_network

# import these packages for Deployer class
#####
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
#####

__all__ = ['generate_jit', 'Deployer']


def generate_jit(pretrained_model: str, h_input: int, w_input: int, outdir: str):
    assert not Path(outdir).exists(), "{} already exists.".format(outdir)
    os.makedirs(outdir, exist_ok=True)

    network = load_network(pretrained_model, cuda=False)
    network.eval()
    jit_script = torch.jit.trace(network, torch.rand(1, 3, h_input, w_input))
    torch.jit.save(jit_script, Path(outdir) / "network_jit.pt")


class Deployer(object):

    def __init__(self, path_jit: str, input_size: tuple, output_size: tuple, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
        # default mean, std: imagenet
        self.model = torch.jit.load(path_jit, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.output_size = output_size
        self.mean = mean
        self.std = std

    def _prepare_input(self, image_path):
        image = Image.open(image_path).convert('RGB')
        t = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])
        image = t(image)
        image = image.unsqueeze(0)
        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        return image

    def _resize_output(self, output):
        if tuple(output.size())[-2:] != self.output_size:
            output = F.interpolate(output, size=self.output_size)
        output = output.data.cpu().numpy()
        pred = np.argmax(output, axis=1)
        pred = np.squeeze(pred)
        return pred

    def inference(self, image_path:str):
        image_tensor = self._prepare_input(image_path)
        with torch.no_grad():
            output = self.model(image_tensor)
        pred = self._resize_output(output)
        return pred
