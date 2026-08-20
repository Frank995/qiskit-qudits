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

from collections.abc import Mapping

from qiskit._accelerate.circuit import QuantumRegister, Qubit

class Layout:
    __slots__ = ("_p2v", "_regs", "_v2p")

    _regs: list[QuantumRegister]
    _p2v: dict[int, Qubit]
    _v2p: dict[Qubit, int]

    def __init__(
        self,
        input_dict: Mapping[Qubit, int] | Mapping[int, Qubit] | None = ...,
    ) -> None: ...
