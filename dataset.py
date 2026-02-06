import os
import random

import cv2
import numpy as np
from prompt_points import label_to_point_prompt_global, label_to_point_prompt_local
from torch.utils.data import Dataset
from collections import defaultdict

import albumentations as alb
from display import show_result_sample_figure
from tqdm import tqdm


class vessel_2d_dataset(Dataset):
    def __init__(
        self,
        fov="3M",
        label_type="LargeVessel",
        prompt_positive_num=-1,
        prompt_negative_num=-1,
        is_local=True,
        is_training=True,
        base_dir="datasets/OCTA-500",
    ):

        self.prompt_positive_num = prompt_positive_num
        self.prompt_negative_num = prompt_negative_num
        self.is_local = is_local
        self.is_training = is_training
        self.counter = 0
        self.img_size = [1024, 1024]

        layers = ["FULL"]
        modal = "OCTA"
        label_dir = os.path.join(base_dir, f"OCTA_{fov}", f"GT_{label_type}")

        filenames = [
            f for f in os.listdir(label_dir) if f.endswith(".png") and f[:-4].isdigit()
        ]
        filenames_sorted = sorted(filenames, key=lambda x: int(x[:-4]))
        self.sample_ids = [x[:-4] for x in filenames_sorted]
        print(f"Total samples found: {len(self.sample_ids)}")

        self.image_paths = []
        for sample_id in tqdm(
            self.sample_ids, desc="Preparing image paths", unit="sample"
        ):
            channel_paths = []
            for layer in layers:
                image_path = os.path.join(
                    base_dir,
                    f"OCTA_{fov}",
                    "ProjectionMaps",
                    f"{modal}({layer})",
                    f"{sample_id}.png",
                )
                channel_paths.append(image_path)
            self.image_paths.append(channel_paths)

        self.label_paths = [
            os.path.join(label_dir, f"{sample_id}.png") for sample_id in self.sample_ids
        ]

        prob = 0.2
        self.transform = alb.Compose(
            [
                alb.Resize(*self.img_size, interpolation=cv2.INTER_NEAREST),
                alb.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
                alb.RandomBrightnessContrast(p=prob),
                alb.CLAHE(p=prob),
                # alb.SafeRotate(limit=15, p=prob),
                alb.ShiftScaleRotate(p=prob),
                alb.VerticalFlip(p=prob),
                alb.HorizontalFlip(p=prob),
                # alb.AdvancedBlur(p=prob),
                # alb.MixUp(p=prob),
                # alb.GridDistortion(p=prob),
                # alb.GridDropout(p=prob),
                alb.ElasticTransform(p=prob),
            ]
        )

    def __len__(self):
        return len(self.sample_ids)

    # def __getitem__(self, index):
    #     # print(index)
    #     indices = [index - 1, index, index + 1]
    #     images_resized = []
    #     for idx in indices:
    #         if idx < 0:
    #             idx = 0
    #         elif idx >= len(self.image_paths):
    #             idx = len(self.image_paths) - 1
    #         img = self._load_image(idx)

    #         img_resized = []
    #         for channel in img:
    #             channel_resized = cv2.resize(
    #                 channel,
    #                 (self.img_size[1], self.img_size[0]),
    #                 interpolation=cv2.INTER_NEAREST,
    #             )
    #             img_resized.append(channel_resized)
    #         img_resized = np.array(img_resized)
    #         images_resized.append(img_resized)

    #     concatenated_image = np.concatenate(images_resized, axis=0)

    #     label = self._load_label(index)
    #     label = cv2.resize(
    #         label, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST
    #     )

        
    #     current_resized_image = images_resized[1]

    #     image, prompt_points, prompt_type, selected_component = self.get_sam_item(
    #         current_resized_image, label
    #     )
    #     return (
    #         image,
    #         selected_component,
    #         self.sample_ids[index],
    #     )
    
    # def __getitem__(self, index):
    #     while True:
    #         indices = [index - 1, index, index + 1]
    #         images_resized = []
    #         for idx in indices:
    #             if idx < 0:
    #                 idx = 0
    #             elif idx >= len(self.image_paths):
    #                 idx = len(self.image_paths) - 1
    #             img = self._load_image(idx)

    #             img_resized = []
    #             for channel in img:
    #                 channel_resized = cv2.resize(
    #                     channel,
    #                     (self.img_size[1], self.img_size[0]),
    #                     interpolation=cv2.INTER_NEAREST,
    #                 )
    #                 img_resized.append(channel_resized)
    #             img_resized = np.array(img_resized)
    #             images_resized.append(img_resized)

    #         label = self._load_label(index)
    #         label = cv2.resize(
    #             label, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST
    #         )

    #         if np.sum(label) == 0:
    #             # print("skip")
    #             index = (index + 1) % len(self)
    #             continue  # 重新加载下一个样本

    #         current_resized_image = images_resized[1]

    #         image, prompt_points, prompt_type, selected_component = self.get_sam_item(
    #             current_resized_image, label
    #         )
    #         return (
    #             image,
    #             selected_component,
    #             self.sample_ids[index],
    #         )

    def __getitem__(self, index):
        # print(index)
        img = self._load_image(index)

        img_resized = []
        for channel in img:
            channel_resized = cv2.resize(
                channel,
                (self.img_size[1], self.img_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            img_resized.append(channel_resized)
        img_resized = np.array(img_resized)

        label = self._load_label(index)
        label = cv2.resize(
            label, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST
        )

        current_resized_image = img_resized
        image, prompt_points, prompt_type, selected_component = self.get_sam_item(
            current_resized_image, label
        )

        return (
            image,
            selected_component,
            self.sample_ids[index],
        )



    def _load_image(self, index):
        image_channels = []
        for path in self.image_paths[index]:
            # print(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {path}")
            image_channels.append(img)
        return np.array(image_channels)

    def _load_label(self, index):
        label_path = self.label_paths[index]
        # print(label_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Label not found: {label_path}")
        return label / 255.0

    def get_sam_item(self, image, label):
        if self.is_training:
            transformed = self.transform(
                **{
                    "image": image.transpose((1, 2, 0)),  # (H, W, C)
                    "mask": label[np.newaxis, :].transpose((1, 2, 0)),
                }
            )
            image, label = (
                transformed["image"].transpose((2, 0, 1)),  # (C, H, W)
                transformed["mask"].transpose((2, 0, 1))[0],
            )

        ppn, pnn = self.prompt_positive_num, self.prompt_negative_num
        if self.is_local:
            random_max = 4
            if ppn == -1:
                ppn = random.randint(0, random_max)
            if pnn == -1:
                pnn = random.randint(int(ppn == 0), random_max)
            (
                selected_component,
                prompt_points_pos,
                prompt_points_neg,
            ) = label_to_point_prompt_local(label, ppn, pnn)
        else:
            (
                selected_component,
                prompt_points_pos,
                prompt_points_neg,
            ) = label_to_point_prompt_global(label, ppn, pnn)

        prompt_type = np.array(
            [1] * len(prompt_points_pos) + [0] * len(prompt_points_neg)
        )

        prompt_points = np.array(prompt_points_pos + prompt_points_neg)

        return image, prompt_points, prompt_type, selected_component


# if __name__=="__main__":
#     dataset = vessel_2d_dataset(is_local=True, prompt_positive_num=1, is_training=True)
#     for image, prompt_points, prompt_type, selected_component, sample_id in dataset:
#         print(np.max(selected_component))
