import os
from typing import Union, List
import torch
import numpy as np
from PIL import Image
from backend.inpaint.utils.lama_util import prepare_img_and_mask
from backend import config


class LamaInpaint:
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_path=None) -> None:
        if model_path is None:
            model_path = os.path.join(config.LAMA_MODEL_PATH, 'big-lama.pt')
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray]):
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_height, orig_width = np.array(image).shape[:2]
        image, mask = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cur_res[:orig_height, :orig_width]
            return cur_res

    def inpaint_batch(self, images: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        img_tensors = []
        mask_tensors = []
        orig_dims = []

        for img in images:
            h, w = img.shape[:2]
            orig_dims.append((h, w))
            img_tensor, mask_tensor = prepare_img_and_mask(img, mask, self.device)
            img_tensors.append(img_tensor)
            mask_tensors.append(mask_tensor)

        img_batch = torch.cat(img_tensors, dim=0)
        mask_batch = torch.cat(mask_tensors, dim=0)

        with torch.inference_mode():
            inpainted_batch = self.model(img_batch, mask_batch)

        output_images = []
        for i in range(inpainted_batch.shape[0]):
            orig_h, orig_w = orig_dims[i]
            cur_res = inpainted_batch[i].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cur_res[:orig_h, :orig_w]
            output_images.append(cur_res)

        return output_images

