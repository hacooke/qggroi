import numpy as np


class ROI:
    def __init__(self, bounds: list[list[int, int]]):
        assert len(bounds) == 2
        assert len(bounds[0]) == 2
        assert len(bounds[1]) == 2
        self.x_bounds = bounds[0]
        self.y_bounds = bounds[1]

    def integral(self, data: np.ndarray) -> float:
        """Calculate the integral of this ROI in the given data array."""
        return np.sum(self.apply_bounds(data))

    def apply_bounds(self, data: np.ndarray) -> np.ndarray:
        return data[self.y_bounds[0] : self.y_bounds[1], self.x_bounds[0] : self.x_bounds[1]]

    def __repr__(self) -> None:
        return repr(self.x_bounds) + repr(self.y_bounds)
