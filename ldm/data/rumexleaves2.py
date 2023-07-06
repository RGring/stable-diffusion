import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
from annotation_converter.AnnotationConverter import AnnotationConverter
from annotation_converter.Annotation import Annotation
from annotation_converter.Polygon import Polygon
from annotation_converter.Polyline import Polyline
import matplotlib.pyplot as plt
import itertools


class RumexLeaves(Dataset):
    def __init__(self, data_root, img_list_file, annotation_file, size=None, random_crop=False, interpolation="bicubic", gen_annotation=False, load_image= True):

        self.root = data_root
        self.annotation_file = f"{self.root}/{annotation_file}"
        with open(img_list_file, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {"black": 3, "background": 0, "leaf_blade": 1, "leaf_stem": 2}

        self.gen_annotation = gen_annotation
        self.load_image = load_image


        # Image Transformations/
        transformations = []
        if size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            
            transformations.append(albumentations.LongestMaxSize(max_size=size, interpolation=self.interpolation, p=1.0))
            transformations.append(albumentations.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, position="center", value=(0, 0, 0), mask_value=self.labels["black"], p=1.0))

        
            center_crop = not random_crop
            if center_crop:
                cropper = albumentations.CenterCrop(height=size, width=size)
            else:
                cropper = albumentations.RandomCrop(height=size, width=size)
            transformations.append(cropper)
        self.preprocessor = albumentations.Compose(transformations,
                                                   keypoint_params=albumentations.KeypointParams(format='xy', remove_invisible=False),
                                                   additional_targets={'polyline_points': 'keypoints'})

    def __len__(self):
        return self._length
    
    def generate_mask(self, mask, polygons_points, polylines_points, scale):
        cv2.fillPoly(mask, polygons_points, self.labels["leaf_blade"])
        cv2.polylines(mask, polylines_points, False, self.labels["leaf_stem"], int(25 * scale))
        return mask
    
    def __getitem__(self, i):
        image_id = self.image_paths[i]
        annotation = AnnotationConverter.read_cvat_by_id(self.annotation_file, image_id)
        if self.load_image:
            image = Image.open(f"{self.root}/{image_id}")
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
        else:
            image = np.zeros((int(annotation.get_img_width()), int(annotation.get_img_height()), 3), dtype=np.uint8)

        polygons = annotation.get_polygons()
        polygons_points = [np.array(polygon.get_polygon_points_as_array()) for polygon in polygons]
        polylines = annotation.get_polylines()
        polylines_points = [np.array(polyline.get_polyline_points_as_array()) for polyline in polylines]

        # Transform
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = self.generate_mask(mask, polygons_points, polylines_points, 1)

        trans = self.preprocessor(image=image,
                                  mask=mask,
                                  keypoints=list(itertools.chain(*polygons_points)), 
                                  polyline_points=list(itertools.chain(*polylines_points)))
        
        # Postprocess transformed labels
        trans_polygons_points = np.zeros((50, 100, 2), np.int32)
        count = 0
        for i, polygon_points in enumerate(polygons_points):
            trans_polygons_points[i, :len(polygon_points), :] = np.array(trans["keypoints"][count:count+len(polygon_points)]).astype(int)
            count += len(polygon_points)
        
        trans_polylines_points = np.zeros((50, 8, 2), np.int32)
        count = 0
        for i, polyline_points in enumerate(polylines_points):
            trans_polylines_points[i, :len(polyline_points), :] = np.array(trans["polyline_points"][count:count+len(polyline_points)]).astype(int)
            count += len(polyline_points)

        image_trans = None
        if True: #self.load_image:
            image_trans = (trans["image"]/127.5 - 1.0).astype(np.float32)
        # scale = (trans["image"].shape[0] * trans["image"].shape[1]) / (image.shape[0] * image.shape[1])
        mask = trans["mask"]
        onehot = np.eye(len(self.labels))[mask]

        # annotation = None
        # if self.gen_annotation:
        #     annotation = self.generate_annotation(trans["image"], trans_polygons_points, trans_polylines_points)
        
        ret = {"image": image_trans,
               "segmentation": onehot,
               "polygons": trans_polygons_points,
               "polylines": trans_polylines_points,
               }
        return ret



class RumexLeavesSegEval(RumexLeaves):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(img_list_file="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_val.txt",
                         data_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
                         annotation_file="annotations.xml",
                         size=size, random_crop=random_crop, interpolation=interpolation)

class RumexLeavesSegTest(RumexLeaves):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(img_list_file="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_test.txt",
                         data_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
                         annotation_file="annotations.xml",
                         size=size, random_crop=random_crop, interpolation=interpolation)
        
class RumexLeavesSeg1Batch(RumexLeaves):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(img_list_file="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_1_batch.txt",
                         data_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
                         annotation_file="annotations.xml",
                         size=size, random_crop=random_crop, interpolation=interpolation)

class RumexLeavesTrain(RumexLeaves):
    def __init__(self, size=None, random_crop=True, interpolation="bicubic"):
        super().__init__(img_list_file="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_train.txt",
                         data_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
                         annotation_file="annotations.xml",
                         size=size, random_crop=random_crop, interpolation=interpolation)

class RumexLeavesGen(RumexLeaves):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(img_list_file="/home/ronja/data/generated/RumexLeaves/composed_annotations/dataset_splits/random_train.txt",
                         data_root="/home/ronja/data/generated/RumexLeaves/composed_annotations/",
                         annotation_file="annotations.xml",
                         size=size, random_crop=False, interpolation=interpolation, load_image=False, gen_annotation=True)

def generate_annotation(image, polygons_points, polylines_points):
    polygons = []
    for polygon_points in polygons_points:
        if sum(sum(polygon_points)) == 0:
            break
        pol = Polygon("leaf_blade")
        pol.set_polygon_points_as_array(polygon_points[np.where(polygon_points!=(0, 0))[0]])
        polygons.append(pol)
    
    polylines = []
    for polyline_points in polylines_points:
        if sum(sum(polyline_points)) == 0:
            break
        pol = Polyline("leaf_stem")
        pol.set_polyline_points_as_array(polyline_points[np.where(polyline_points!=(0, 0))[0]])
        polylines.append(pol)
    annotation = Annotation("str", float(image.shape[0]), float(image.shape[1]), polygon_list=polygons, polyline_list=polylines)
    return annotation


def main():
    # ds = RumexLeaves(root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
    #                  img_list_file="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_1_batch.txt",
    #                  annotation_file="annotations.xml",
    #                  size=512,
    #                  gen_annotation=True,
    #                  )
    
    # ds = RumexLeavesGen(size=512)
    ds = RumexLeavesTrain(size=512)
    
    for i in range(1):
        ret = ds[i]
        image, mask, polygons, polylines = ret["image"], ret["segmentation"], ret["polygons"], ret["polylines"]
        annotation = generate_annotation(image, polygons, polylines)

        AnnotationConverter.write_cvat([annotation], "test.xml")

        polygons = annotation.get_polygons()
        polygons_points = [np.array(polygon.get_polygon_points_as_array()) for polygon in polygons]
        cv2.polylines(image, polygons_points, True, (255, 0, 0), 5)

        polylines = annotation.get_polylines()
        polylines_points = [np.array(polyline.get_polyline_points_as_array()) for polyline in polylines]    
        cv2.polylines(image, polylines_points, False, (0, 255, 0), 5)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(mask)
        plt.show()

if __name__ == "__main__":
    main()