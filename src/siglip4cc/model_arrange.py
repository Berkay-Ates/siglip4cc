import argparse
import os
from pathlib import Path

import torch
from PIL.Image import Image
from torch.utils.data import DataLoader

from siglip4cc.data_loader import Siglip_DataLoader
from siglip4cc.siglip.modeling_siglip_txt import SiglipTextConfig
from siglip4cc.siglip.modeling_siglip_view import SiglipVisionConfig
from siglip4cc.modeling_siglip import Siglip4IDC


def get_model_args():
    return {
        "text_config": str(SiglipTextConfig()),
        "vision_config": str(SiglipVisionConfig()),
    }


def encode_rgb_images_for_rsformer(
    model: Siglip4IDC,
    before_image: Image | Path,
    after_image: Image | Path,
    device: torch.device,
) -> torch.Tensor:
    dataset = Siglip_DataLoader(
        bef_img_path=before_image,
        aft_img_path=after_image,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        batch = tuple(t.to(device) for t in next(iter(dataloader)))
        (bef_image, aft_image, image_mask) = batch

        image_pair = torch.cat([bef_image, aft_image], 1)

        visual_output, visual_hidden = model.get_visual_output(image_pair, image_mask)

    return visual_hidden


def load_model(model_file: str | os.PathLike) -> Siglip4IDC:
    if os.path.exists(model_file):
        print("Model loaded from %s", model_file)

        model = Siglip4IDC.from_pretrained(pretrained_model_path=model_file)
    else:
        raise FileNotFoundError(f"The path doesn't exists: {model_file}")

    return model.eval()
