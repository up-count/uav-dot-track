import numpy as np
import cv2
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 12 * 12, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = self.fc(x)
        return x

class Classificator(nn.Module):
    def __init__(self, model_path):
        super(Classificator, self).__init__()
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        print(f'[LOGS] Classificator model loaded from {model_path}')

    def predict(self, frame, xyr):
        x, y, r = xyr

        if r < 5 or r < 5:
            return False, 0.0

        x1 = int(max(0, x-r))
        y1 = int(max(0, y-r))
        x2 = int(min(frame.shape[1], x+r))
        y2 = int(min(frame.shape[0], y+r))

        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = np.transpose(roi, (2, 0, 1))
        roi = np.expand_dims(roi, 0)

        with torch.no_grad():
            output = self.model(torch.tensor(roi).float())
            output = torch.sigmoid(output)
            prob = output[0][0].item()

        return True, prob
