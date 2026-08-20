# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root
# directory directory of this source tree or at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections.abc import Sequence
from typing import Self

class OpShape:
    _num_qargs_l: int
    _num_qargs_r: int
    _dims_l: tuple[int, ...] | None
    _dims_r: tuple[int, ...] | None

    def __init__(
        self,
        dims_l: Sequence[int] | None = ...,
        dims_r: Sequence[int] | None = ...,
        num_qargs_l: int | None = ...,
        num_qargs_r: int | None = ...,
    ) -> None: ...
    @property
    def settings(self) -> dict[str, tuple[int, ...] | int | None]: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def copy(self) -> OpShape: ...
    @property
    def size(self) -> int: ...
    @property
    def num_qubits(self) -> int | None: ...
    @property
    def num_qargs(self) -> tuple[int, int]: ...
    @property
    def shape(self) -> tuple[int, int] | tuple[int]: ...
    @property
    def tensor_shape(self) -> tuple[int, ...]: ...
    @property
    def is_square(self) -> bool: ...
    def dims_r(self, qargs: Sequence[int] | None = ...) -> tuple[int, ...]: ...
    def dims_l(self, qargs: Sequence[int] | None = ...) -> tuple[int, ...]: ...
    @property
    def _dim_r(self) -> int: ...
    @property
    def _dim_l(self) -> int: ...
    def validate_shape(self, shape: Sequence[int]) -> bool: ...
    def _validate(
        self,
        shape: Sequence[int],
        raise_exception: bool = ...,
    ) -> bool: ...
    @classmethod
    def auto(
        cls,
        shape: Sequence[int] | None = ...,
        dims_l: int | Sequence[int] | None = ...,
        dims_r: int | Sequence[int] | None = ...,
        dims: int | Sequence[int] | None = ...,
        num_qubits_l: int | None = ...,
        num_qubits_r: int | None = ...,
        num_qubits: int | None = ...,
    ) -> OpShape: ...
    def subset(
        self,
        qargs: int | Sequence[int] | None = ...,
        qargs_l: int | Sequence[int] | None = ...,
        qargs_r: int | Sequence[int] | None = ...,
    ) -> OpShape: ...
    def remove(
        self,
        qargs: int | Sequence[int] | None = ...,
        qargs_l: int | Sequence[int] | None = ...,
        qargs_r: int | Sequence[int] | None = ...,
    ) -> OpShape: ...
    def reverse(self) -> OpShape: ...
    def transpose(self) -> OpShape: ...
    def tensor(self, other: OpShape) -> OpShape: ...
    def expand(self, other: OpShape) -> OpShape: ...
    @classmethod
    def _tensor(cls, a: OpShape, b: OpShape) -> OpShape: ...
    def compose(
        self,
        other: OpShape,
        qargs: Sequence[int] | None = ...,
        front: bool = ...,
    ) -> OpShape: ...
    def dot(
        self,
        other: OpShape,
        qargs: Sequence[int] | None = ...,
    ) -> OpShape: ...
    def _validate_add(
        self,
        other: OpShape,
        qargs: Sequence[int] | None = ...,
    ) -> Self: ...
