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

from abc import ABC

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import (
    AdjointMixin,
    LinearMixin,
    TolerancesMixin,
)

class LinearOp(
    BaseOperator,
    AdjointMixin,
    LinearMixin,
    TolerancesMixin,
    ABC,
): ...
