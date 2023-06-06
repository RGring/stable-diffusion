import torch
import numpy as np
import os
from PIL import Image
import glob

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
# from torchvision.utils import save_image
from einops import rearrange

from ldm.data.rumexleaves import RumexLeavesSegEval
from annotation_converter.AnnotationConverter import AnnotationConverter


global i_global
i_global = 0

def save_image(imgs, output_path, rescale=True):
    samples = imgs
    samples = torch.clamp(samples, -1., 1.)
    if rescale:
        samples = (samples + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    for i_sample in range(samples.shape[0]):
        sample = samples[i_sample, :, :, :].transpose(0, 1).transpose(1, 2)
        sample = sample.numpy()
        sample = (sample * 255).astype(np.uint8)
        Image.fromarray(sample).save( f"{output_path}/{i_global + i_sample}.png")

def sample_rumex_leaves(config_path, ckpt_path, output_path, dataset, batch_size, num_imgs=10, annotation_file=None):
    global i_global

    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    annotations = []

    for i in range(int(num_imgs/batch_size)):
        x = next(iter(dataloader))
        # Extract original annotations
        if annotation_file is not None:
            for i_image_file, image_file in enumerate(x["relative_file_path_"]):
                annotation = AnnotationConverter.read_cvat_by_id(annotation_file, image_file)
                annotation.image_name = f"{i_global + i_image_file}.png"
                annotations.append(annotation)
                          
        seg = x['segmentation']
        with torch.no_grad():
            seg = rearrange(seg, 'b h w c -> b c h w')
            condition = model.to_rgb(seg)

            seg = seg.to('cuda').float()
            seg = model.get_learned_conditioning(seg)
            sample, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                            ddim_steps=200, eta=1.0, quantize_denoised=False)

            sample = model.decode_first_stage(sample)
            sample = sample.detach().cpu()
            save_image(sample, f"{output_path}/images")
            save_image(condition, f"{output_path}/masks")
            i_global += batch_size
            
    # Save annotations
    if annotation_file is not None:
        AnnotationConverter.write_cvat(annotations, f"{output_path}/{os.path.basename(annotation_file)}")
        

if __name__ == '__main__':
    log_folder = 'logs/ldm/2023-05-10T14-48-31_rumexleaves-ldm-vq-4_pretr1'
    config_path = glob.glob(f'{log_folder}/configs/*-project.yaml')[0]
    # If we want to trace back segmentation conditions to original annotations, provide appropriate annotation file
    orig_annotation_file = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/annotations_oriented_bb.xml"

    num_imgs = 10
    batchsize = 10

    ckpt_paths = [f'{log_folder}/checkpoints/epoch=000099.ckpt',
                 f'{log_folder}/checkpoints/epoch=000199.ckpt',
                 f'{log_folder}/checkpoints/epoch=000599.ckpt',
                 f'{log_folder}/checkpoints/epoch=000999.ckpt']
    
    for ckpt_path in ckpt_paths:
        output_path = (f"/home/ronja/data/generated/test/{os.path.basename(ckpt_path)}").replace(".ckpt", "")

        os.makedirs(f"{output_path}/images", exist_ok=True)
        os.makedirs(f"{output_path}/masks", exist_ok=True)
        dataset = RumexLeavesSegEval(size=256)

        sample_rumex_leaves(config_path, ckpt_path, output_path, dataset, batch_size, num_imgs, orig_annotation_file)