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

class GroupMixin(ABC):
    def __and__(self, other: object) -> Self: ...
    def __power__(self, other: object) -> Self: ...
    def __xor__(self, other: object) -> Self: ...
    def __matmul__(self, other: object) -> Self: ...
    @abstractmethod
    def tensor(self, other: object) -> Self: ...
    @abstractmethod
    def expand(self, other: object) -> Self: ...
    @abstractmethod
    def compose(
        self,
        other: object,
        qargs: Sequence[int] | None = ...,
        front: bool = ...,
    ) -> Self: ...
    def dot(
        self,
        other: object,
        qargs: Sequence[int] | None = ...,
    ) -> Self: ...
    def power(self, n: int) -> Self: ...
