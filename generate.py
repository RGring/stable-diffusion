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

from ldm.data.rumexleaves2 import RumexLeavesGen
from annotation_converter.AnnotationConverter import AnnotationConverter
from annotation_converter.Annotation import Annotation
from annotation_converter.Polygon import Polygon
from annotation_converter.Polyline import Polyline


class RumexLeavesGenerator:
    def __init__(self, log_folder) -> None:
        self.i_global = 0
        self.log_folder = log_folder
        config_path = glob.glob(f'{self.log_folder}/configs/*-project.yaml')[0]
        self.config = OmegaConf.load(config_path)

        self.batch_size = 5

    def generate(self, num_samples, img_size, ckpt_path, out_path):
        dataset = RumexLeavesGen(size=img_size)
        output_path = (f"{out_path}/{os.path.basename(log_folder)}/{os.path.basename(ckpt_path)}").replace(".ckpt", "")
        os.makedirs(f"{output_path}/images", exist_ok=True)
        os.makedirs(f"{output_path}/masks", exist_ok=True)
        self.sample_rumex_leaves(ckpt_path, output_path, dataset, num_samples)

    def sample_rumex_leaves(self, ckpt_path, output_path, dataset, num_samples=10):
        model, _ = load_model(self.config, ckpt_path, None, None)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(int(num_samples/self.batch_size)):
            x = next(iter(dataloader))
                            
            seg = x['segmentation']
            with torch.no_grad():
                seg = rearrange(seg, 'b h w c -> b c h w')
                condition = model.to_rgb(seg)
                seg = seg.to('cuda').float()
                seg = model.get_learned_conditioning(seg)
                self.save_annotation(condition, x['polygons'], x['polylines'], f"{output_path}/images/annotations.xml")
                with model.ema_scope("Plotting"):
                    sample, _ = model.sample_log(cond=seg, batch_size=self.batch_size, ddim=True,
                                                    ddim_steps=200, eta=1.0, quantize_denoised=False)

                    sample = model.decode_first_stage(sample)
                    sample = sample.detach().cpu()
                self.save_image(sample, f"{output_path}/images")
                self.save_image(condition, f"{output_path}/masks")

                self.i_global += self.batch_size
    
    def save_image(self, imgs, output_path, rescale=True):
        samples = imgs
        samples = torch.clamp(samples, -1., 1.)
        if rescale:
            samples = (samples + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        for i_sample in range(samples.shape[0]):
            sample = samples[i_sample, :, :, :].transpose(0, 1).transpose(1, 2)
            sample = sample.numpy()
            sample = (sample * 255).astype(np.uint8)
            Image.fromarray(sample).save( f"{output_path}/{self.i_global + i_sample}.png")
    
    def save_annotation(self, imgs, polygon_batch, polyline_batch, annotation_file):
        for i_sample in range(imgs.shape[0]):
            annotation = self.gen_annotation(imgs[i_sample], polygon_batch[i_sample], polyline_batch[i_sample], i_sample)
            AnnotationConverter.extend_cvat(annotation, annotation_file)
    
    def gen_annotation(self, image, polygons_points, polylines_points, i_sample):
        polygons = []
        polygons_points = polygons_points.cpu().numpy()
        for polygon_points in polygons_points:
            if sum(sum(polygon_points)) == 0:
                break
            pol = Polygon("leaf_blade")
            pol.set_polygon_points_as_array(polygon_points[np.where(np.sum(polygon_points, axis=1) != 0)])
            polygons.append(pol)
        
        polylines = []
        polylines_points = polylines_points.cpu().numpy()
        for polyline_points in polylines_points:
            if sum(sum(polyline_points)) == 0:
                break
            pol = Polyline("leaf_stem")
            pol.set_polyline_points_as_array(polyline_points[np.where(np.sum(polyline_points, axis=1) != 0)])
            polylines.append(pol)
        annotation = Annotation(f"{self.i_global + i_sample}.png", image.shape[1], image.shape[2], polygon_list=polygons, polyline_list=polylines)
        return annotation


if __name__ == '__main__':
    log_folder = 'logs/ldm/2023-05-10T14-48-31_rumexleaves-ldm-vq-4_pretr1'
    # log_folder = '/home/ronja/log/ldm/2023-07-06T15-17-12_rumexleaves-ldm-vq-4_pretr1_3_frac01'

    out_path = "/home/ronja/data/generated/RumexLeaves"
    img_size = 256
    num_samples = 20

    generator = RumexLeavesGenerator(log_folder)

    ckpt_paths = [
            #  f'{log_folder}/checkpoints/epoch=000049.ckpt',
                # f'{log_folder}/checkpoints/epoch=000099.ckpt',
                f'{log_folder}/checkpoints/epoch=000199.ckpt',
                # f'{log_folder}/checkpoints/epoch=000299.ckpt',
                # f'{log_folder}/checkpoints/epoch=000399.ckpt',
                f'{log_folder}/checkpoints/epoch=000599.ckpt',
                ]
    
    for ckpt_path in ckpt_paths:
        generator.generate(num_samples, img_size, ckpt_path, out_path)
