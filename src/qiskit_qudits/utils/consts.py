"""Utility constants for the repository."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np

IntLike: TypeAlias = int | np.integer[Any]
FloatLike: TypeAlias = IntLike | float | np.floating[Any]
VectorLike: TypeAlias = Sequence[Any] | np.typing.NDArray[Any]

# A qudit must have dimension at least 2; a single-level qudit is
# trivial (it carries no quantum information).
MIN_QUDIT_DIM = 2
