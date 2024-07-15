import argparse
from dataclasses import dataclass

import torch
from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


class Transform:
    def __init__(self, img_size: tuple[int]) -> None:
        self._trans = SceneTextDataModule.get_transform(img_size)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._trans(img)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        help="Model checkpoint (or 'pretrained=<model_id>')",
        default=None,
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--cased", action="store_true", default=False, help="Cased comparison"
    )
    parser.add_argument(
        "--punctuation", action="store_true", default=False, help="Check punctuation"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        default=False,
        help="Evaluate on new benchmark datasets",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="Angle of rotation (counter clockwise) in degrees.",
    )
    parser.add_argument("--device", default="cuda")
    args, unknown = parser.parse_known_args()

    if args.checkpoint is None:
        args.checkpoint = "outputs/parseq/2024-07-09_13-16-48/checkpoints/epoch=44-step=702751-val_accuracy=8.5000-val_NED=22.9592.ckpt"
    kwargs = parse_model_args(unknown)
    print(f"Additional keyword arguments: {kwargs}")

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to('cuda:1')
    hp = model.hparams

    trans = Transform(img_size=hp.img_size)
    img = Image.open("assets/test/en.png").convert('RGB')
    img = trans(img)
    img = img.unsqueeze(0)

    res = model.test_step((img.to(model.device), None), -1)
    print(f"pred:\n{res.confidence}: {res.preds}")


if __name__ == "__main__":
    main()
