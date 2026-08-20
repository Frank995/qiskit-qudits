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

from qiskit._accelerate.circuit import StandardInstructionType
from qiskit.circuit.instruction import Instruction

class Barrier(Instruction):
    _directive: bool
    _standard_instruction_type: StandardInstructionType

    def __init__(self, num_qubits: int, label: str | None = ...) -> None: ...
    def inverse(self, annotated: bool = ...) -> Barrier: ...
