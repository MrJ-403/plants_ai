import cv2
import numpy as np
import torch
import os
from ultralytics import FastSAM

model = FastSAM("FastSAM-s.pt")

def clean_and_isolate_leaf(img, verbose=False) -> cv2.typing.MatLike:
    """
    Clean the image's background and return the leaf with a black background.

    **Note:**
        The image must contain one leaf only, otherwise the model will return other objects or ignore the rest of the leaves.

    :param `img`: the image that contains one leaf.
    :type `img`: MatLike
    :param `verbose`: decides if you want the model to be verbose about the logs or silent.
    :type `verbose`: Boolean
    :returns: object representing the image with a black background.
    :rtype: MatLike

    """

    H, W, _ = img.shape

    # Run inference
    results = model(img, device="cuda" if torch.cuda.is_available() else "cpu", imgsz=256, conf=0.85, iou=0.9, verbose=verbose)

    if len(results) > 0 and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # (N, H_mask, W_mask)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Find the largest detected object by bounding box area
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        best_idx = np.argmax(areas)

        # Extract and resize mask back to the EXACT original image dimensions (H, W)
        mask = masks[best_idx]
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask (0 or 255)
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Apply mask to set background to solid black (0)
        isolated_leaf = cv2.bitwise_and(img, img, mask=binary_mask)

        return isolated_leaf
    else:
        # If the model fails to detect objects, we return the original image as it is
        return img
