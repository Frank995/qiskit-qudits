from typing import Any, Literal

import numpy as np

def dft(
    n: int,
    scale: Literal["sqrtn", "n"] | None = None,
) -> np.typing.NDArray[Any]: ...
