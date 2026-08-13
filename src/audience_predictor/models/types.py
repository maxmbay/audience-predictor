from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class TitleReps:
    z: FloatArray
    bias: FloatArray
    movie_row: IntArray


@dataclass
class TitlePosterior:
    mu: FloatArray
    var: FloatArray
    bias: FloatArray
    movie_row: IntArray

    def mean_reps(self) -> TitleReps:
        return TitleReps(z=self.mu, bias=self.bias, movie_row=self.movie_row)
