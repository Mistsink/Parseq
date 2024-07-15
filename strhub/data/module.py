from pathlib import PurePath
from typing import Callable, Optional, Sequence
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl

from .dataset_with_gen import GenDataset
from .custom_generator import Generator, collate_text_label


class SceneTextDataModule(pl.LightningDataModule):
    # TEST_BENCHMARK_SUB = ("IIIT5k", "SVT", "IC13_857", "IC15_1811", "SVTP", "CUTE80")
    # TEST_BENCHMARK = ("IIIT5k", "SVT", "IC13_1015", "IC15_2077", "SVTP", "CUTE80")
    # TEST_NEW = ("ArT", "COCOv1.4", "Uber")
    # TEST_ALL = tuple(set(TEST_BENCHMARK_SUB + TEST_BENCHMARK + TEST_NEW))

    def __init__(
        self,
        root_dir: str,
        train_dir: str,
        img_size: Sequence[int],
        max_label_length: int,
        charset_path: str,
        batch_size: int,
        num_workers: int,
        augment: bool,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        min_image_dim: int = 0,
        rotation: int = 0,
        collate_fn: Optional[Callable] = collate_text_label,

        train_sample_num: int = 100000,
        eval_sample_num: int = -1,
        pred_sample_num: int = -1,
    ):
        super().__init__()

        self.img_size = tuple(img_size)
        # self.max_label_length = max_label_length

        # self.root_dir = root_dir
        # self.train_dir = train_dir
        # self.charset_train = charset_path
        # self.charset_test = charset_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        # self.remove_whitespace = remove_whitespace
        # self.normalize_unicode = normalize_unicode
        # self.min_image_dim = min_image_dim
        # self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None

        self.train_sample_num = train_sample_num
        self.eval_sample_num = eval_sample_num
        self.pred_sample_num = pred_sample_num

    def set_generators(
        self,
        train_generator: Generator,
        eval_generator: Optional[Generator] = None,
        pred_generator: Optional[Generator] = None,
    ):
        if self.train_sample_num == -1:
            self.train_sample_num = len(train_generator)
        if self.eval_sample_num == -1 and eval_generator is not None:
            self.eval_sample_num = len(eval_generator)
        if self.pred_sample_num == -1 and pred_generator is not None:
            self.pred_sample_num = len(pred_generator)

        self.train_generator = train_generator
        self.eval_generator = eval_generator
        self.pred_generator = pred_generator

    @staticmethod
    def get_transform(img_size: tuple[int], augment: bool = False, rotation: int = 0):
        transforms = []
        if augment:
            from .augment import rand_augment_transform

            transforms.append(rand_augment_transform())
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        transforms.extend(
            [
                # T.Resize(img_size, T.InterpolationMode.BICUBIC),
                lambda img: SceneTextDataModule.pad_to_aspect_ratio(img, img_size, fill=(255, 255, 255)),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )
        return T.Compose(transforms)

    @staticmethod
    def pad_to_aspect_ratio(img: Image.Image, target_size: tuple[int, int], fill: tuple[int, int, int] = (255, 255, 255)):
        # 获取目标宽高比
        target_ratio = target_size[0] / target_size[1]
        # 获取图像的实际宽高比
        img_ratio = img.width / img.height
        
        if img_ratio > target_ratio:
            new_width = target_size[0]
            new_height = int(new_width / img_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * img_ratio)
        
        # 缩放图像
        img = img.resize((new_width, new_height), Image.BICUBIC)
        
        # 创建新图像并将缩放后的图像粘贴进去
        new_img = Image.new("RGB", target_size, fill)
        new_img.paste(img, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))
        
        return new_img

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment)
            # root = PurePath(self.root_dir, "train", self.train_dir)
            # self._train_dataset = build_tree_dataset(
            #     root,
            #     self.charset_train,
            #     self.max_label_length,
            #     self.min_image_dim,
            #     self.remove_whitespace,
            #     self.normalize_unicode,
            #     transform=transform,
            # )
            self._train_dataset = GenDataset(self.train_sample_num, transform=transform, generator=self.train_generator)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transform = self.get_transform(self.img_size)
            self._val_dataset = GenDataset(self.eval_sample_num, transform=transform, generator=self.eval_generator)
            # root = PurePath(self.root_dir, "val")
            # self._val_dataset = build_tree_dataset(
            #     root,
            #     self.charset_test,
            #     self.max_label_length,
            #     self.min_image_dim,
            #     self.remove_whitespace,
            #     self.normalize_unicode,
            #     transform=transform,
            # )
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    # def test_dataloaders(self, subset):
    #     transform = self.get_transform(self.img_size, rotation=self.rotation)
    #     root = PurePath(self.root_dir, "test")
    #     datasets = {
    #         s: LmdbDataset(
    #             str(root / s),
    #             self.charset_test,
    #             self.max_label_length,
    #             self.min_image_dim,
    #             self.remove_whitespace,
    #             self.normalize_unicode,
    #             transform=transform,
    #         )
    #         for s in subset
    #     }
    #     return {
    #         k: DataLoader(
    #             v,
    #             batch_size=self.batch_size,
    #             num_workers=self.num_workers,
    #             pin_memory=True,
    #             collate_fn=self.collate_fn,
    #         )
    #         for k, v in datasets.items()
    #     }
