import torch
import numpy as np
from PIL import Image

# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoProcessor

# pip install opencv-python
import cv2


class RawImageExtractorCV2:
    def __init__(self, centercrop=False, size=224):
        self.centercrop = centercrop
        self.size = size
        self.transform = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    def load_image(self, data):

        if isinstance(data, Image.Image):
            return self.get_image_data(data)

        elif isinstance(data, (str, Path)):
            return self.get_image_data_path(data)

        elif isinstance(data, torch.Tensor):  # Eğer Tensor ise
            return self.get_image_data_tensor(data)

    def image_to_tensor(self, image_file, preprocess):
        image = Image.open(image_file).convert("RGB")
        image_data = preprocess(images=image, return_tensors="pt")

        return {"image": image_data}

    def get_image_data(self, image_path):
        image_loaded = image_loaded.convert("RGB")
        image_loaded = self.transform(image_loaded)

        return {"image": image_loaded}

    def get_image_data_path(self, image_path):
        image_input = self.image_to_tensor(image_path, self.transform)
        return image_input

    def get_image_data_tensor(self, image_tensor):
        numpy_img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        image = Image.fromarray(numpy_img)

        # PIL görüntüsüne çevir
        return self.get_image_data(image)


# An ordinary video frame extractor based CV2
RawImageExtractor = RawImageExtractorCV2
