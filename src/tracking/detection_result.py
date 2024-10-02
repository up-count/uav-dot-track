from dataclasses import dataclass
from typing import Tuple


@dataclass
class DetectionResult:
    """
    A class to represent the result of a detection.

    Attributes
    ----------
    label : str
        The label of the detected object.
    confidence : float
        The confidence of the detection.
    x : int
        The x coordinate of the center of the detected object.
    y : int
        The y coordinate of the center of the detected object.
    w : int
        The width of the detected object.
    h : int
        The height of the detected object.
    """

    label: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    
    @property
    def xyxy(self):
        return (self.x - self.w//2, self.y - self.h//2, self.x + self.w//2, self.y + self.h//2)
    
    @property
    def xywh(self):
        return (self.x, self.y, self.w, self.h)


def from_numpy_to_detection_results(predictions):
    """
    Convert a numpy array to a list of DetectionResult objects.

    Parameters
    ----------
    predictions : np.ndarray
        The predictions to convert. [N, 3] -> [X, Y, confidence]

    Returns
    -------
    List[DetectionResult]
        The list of DetectionResult objects.
    """
    return [DetectionResult(label='', confidence=p[2], x=p[0], y=p[1], w=20, h=20) for p in predictions]
