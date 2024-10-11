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


def from_numpy_to_detection_results(predictions, alt):
    """
    Convert a numpy array to a list of DetectionResult objects.

    Parameters
    ----------
    predictions : np.ndarray
        The predictions to convert. [N, 3] -> [X, Y, confidence]
    alt: float
        The altitude of the frame.

    Returns
    -------
    List[DetectionResult]
        The list of DetectionResult objects.
    """
    if alt < 0:
        size = 20
    else:
        size = 100 / alt * 20

    return [DetectionResult(label='', confidence=p[2], x=p[0], y=p[1], w=size, h=size) for p in predictions]
