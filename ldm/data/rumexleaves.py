import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=182, shift_segmentation=True,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.segmentation_root, l.replace(".jpg", ".png"))
                                   for l in self.image_paths]
        }

        size = None if size is not None and size<=0 else size
        self.size = size

        transformations = []
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            

            transformations.append(albumentations.LongestMaxSize(max_size=self.size, interpolation=self.interpolation, p=1.0))
            transformations.append(albumentations.PadIfNeeded(min_height=self.size, min_width=self.size, border_mode=cv2.BORDER_CONSTANT, position="center", value=(0, 0, 0), mask_value=0, p=1.0))
            
            self.center_crop = not random_crop
            if self.center_crop:
                transformations.append(albumentations.CenterCrop(height=self.size, width=self.size))
            else:
                transformations.append(albumentations.RandomCrop(height=self.size, width=self.size))
        self.preprocessor = albumentations.Compose(transformations)
                                                   

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        segmentation = Image.open(example["segmentation_path_"])
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.shift_segmentation:
            # used to support segmentations containing unlabeled==255 label
            segmentation = segmentation+1


        if self.size is not None:
            processed = self.preprocessor(image=image, mask=segmentation)

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation]
        example["segmentation"] = onehot
        return example


class RumexLeavesSegEval(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_val.txt",
                         data_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
                         segmentation_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/segmentations",
                         size=size, random_crop=random_crop, interpolation=interpolation, n_labels=3, shift_segmentation=False)

# class RumexLeavesSegTest(SegmentationBase):
#     def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_csv="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_test.txt",
#                          data_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
#                          segmentation_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/segmentations",
#                          size=size, random_crop=random_crop, interpolation=interpolation, n_labels=4)
        
# class RumexLeavesSeg1Batch(SegmentationBase):
#     def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_csv="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_1_batch.txt",
#                          data_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
#                          segmentation_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/segmentations",
#                          size=size, random_crop=random_crop, interpolation=interpolation, n_labels=4)

# class RumexLeavesSegTrain(SegmentationBase):
#     def __init__(self, size=None, random_crop=True, interpolation="bicubic"):
#         super().__init__(data_csv="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_train.txt",
#                          data_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist",
#                          segmentation_root="/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/segmentations",
#                          size=size, random_crop=random_crop, interpolation=interpolation, n_labels=4)
