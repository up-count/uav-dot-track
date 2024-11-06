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
    r : int
        The radius of the detected object.
    frame_shape : Tuple[int, int]
        The shape of the frame.
    """

    label: str
    confidence: float
    x: int
    y: int
    r: int
    frame_shape: Tuple[int, int]
    from_pointflow: bool = False

    @property
    def xyr(self):
        return int(self.x), int(self.y), int(self.r)


def from_numpy_to_detection_results(predictions, alt, frame_shape):
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
        size = int(100 / alt * 20)

    return [DetectionResult(label='', confidence=p[2], x=p[0], y=p[1], r=size//2, frame_shape=frame_shape) for p in predictions]
