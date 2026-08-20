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

from collections.abc import Iterator
from typing import Any

import numpy as np
from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.instruction import Instruction

class Gate(Instruction):
    __array_priority__: int

    def __init__(
        self,
        name: str,
        num_qubits: int,
        params: list[Any],
        label: str | None = ...,
    ) -> None: ...
    def to_matrix(self) -> np.ndarray: ...
    def power(
        self,
        exponent: float,
        annotated: bool = ...,
    ) -> Gate | AnnotatedOperation: ...
    def __pow__(self, exponent: float) -> Gate: ...
    def _return_repeat(self, exponent: float) -> Gate: ...
    def control(
        self,
        num_ctrl_qubits: int = ...,
        label: str | None = ...,
        ctrl_state: int | str | None = ...,
        annotated: bool | None = ...,
    ) -> Gate | AnnotatedOperation: ...
    @staticmethod
    def _broadcast_single_argument(
        qarg: list[Any],
    ) -> Iterator[tuple[list[Any], list[Any]]]: ...
    @staticmethod
    def _broadcast_2_arguments(
        qarg0: list[Any],
        qarg1: list[Any],
    ) -> Iterator[tuple[list[Any], list[Any]]]: ...
    @staticmethod
    def _broadcast_3_or_more_args(
        qargs: list[Any],
    ) -> Iterator[tuple[list[Any], list[Any]]]: ...
    def broadcast_arguments(
        self,
        qargs: list[Any],
        cargs: list[Any],
    ) -> Iterator[tuple[list[Any], list[Any]]]: ...
    def validate_parameter(self, parameter: Any) -> Any: ...
