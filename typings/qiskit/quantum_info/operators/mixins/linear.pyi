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
from collections.abc import Sequence
from typing import Self

from qiskit.quantum_info.operators.mixins.multiply import MultiplyMixin

class LinearMixin(MultiplyMixin, ABC):
    def __add__(self, other: Self) -> Self: ...
    def __radd__(self, other: Self) -> Self: ...
    def __sub__(self, other: Self) -> Self: ...
    def __rsub__(self, other: Self) -> Self: ...
    @abstractmethod
    def _add(self, other: Self, qargs: Sequence[int] | None = ...) -> Self: ...
