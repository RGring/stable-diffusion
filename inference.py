import torch
import numpy as np
import os
from PIL import Image
import glob

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
# from torchvision.utils import save_image
from einops import rearrange
import torchvision

from ldm.data.landscapes import LandscapesSegEval
from ldm.data.rumexleaves import RumexLeavesSegEval, RumexLeavesSegTrain, RumexLeavesSegTest


def save_image(img, path, rescale=True):
    samples = img
    samples = torch.clamp(img, -1., 1.)
    grid = torchvision.utils.make_grid(samples, nrow=4)
    if rescale:
        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(path)


def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    image_path = f"{os.path.dirname(ckpt_path)}/../images/{os.path.basename(ckpt_path).replace('.ckpt', '')}/val"
    os.makedirs(image_path, exist_ok=True)

    x = next(iter(dataloader))

    seg = x['segmentation']
    real_images = x['image']
    real_images = rearrange(real_images, 'b h w c -> b c h w')

    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        for i in range(5):
            with model.ema_scope("Plotting"):
                samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                            ddim_steps=200, eta=1.0, quantize_denoised=False)

                samples = model.decode_first_stage(samples)
                samples = samples.detach().cpu()
            path = os.path.join(image_path, f"{i}.png")
            save_image(samples, path)

    save_image(condition, f'{image_path}/cond.png')
    save_image(real_images, f'{image_path}/real.png')


if __name__ == '__main__':

    log_folder = '/home/ronja/log/ldm/2023-05-10T14-48-31_rumexleaves-ldm-vq-4_pretr1'
    config_path = glob.glob(f'{log_folder}/configs/*-project.yaml')[0]
    ckpts = glob.glob(f'{log_folder}/checkpoints/*.ckpt')
    ckpts = [
        # f'{log_folder}/checkpoints/epoch=000099.ckpt',
        f'{log_folder}/checkpoints/epoch=000199.ckpt',
        ]

    # dataset = LandscapesSegEval(size=256)
    # dataset = RumexLeavesSegTrain(size=256)
    dataset = RumexLeavesSegEval(size=256)

    for ckpt in ckpts:
        ckpt_path = ckpt
        ldm_cond_sample(config_path, ckpt_path, dataset, 4)