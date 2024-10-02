import cv2
import numpy as np
import torch

def preprocess_image(
        frame: np.ndarray,
        image_size: tuple,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> torch.tensor:
    
    frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_size[0], image_size[1]))
    frame = frame.astype(np.float32) / 255.
    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1)
    
    return torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)
