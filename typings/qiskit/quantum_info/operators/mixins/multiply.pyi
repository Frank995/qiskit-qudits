# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from abc import ABC, abstractmethod
from typing import Self

class MultiplyMixin(ABC):
    def __rmul__(self, other: complex) -> Self: ...
    def __mul__(self, other: complex) -> Self: ...
    def __truediv__(self, other: complex) -> Self: ...
    def __neg__(self) -> Self: ...
    @abstractmethod
    def _multiply(self, other: complex) -> Self: ...
