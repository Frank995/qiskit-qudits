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

from abc import ABCMeta
from typing import Any

class TolerancesMeta(ABCMeta):
    def __init__(cls, *args: Any, **kwargs: Any) -> None: ...
    def _check_value(cls, value: float, value_name: str) -> None: ...
    @property
    def atol(cls) -> float: ...
    @atol.setter
    def atol(cls, value: float) -> None: ...
    @property
    def rtol(cls) -> float: ...
    @rtol.setter
    def rtol(cls, value: float) -> None: ...

class TolerancesMixin(metaclass=TolerancesMeta):
    @property
    def atol(self) -> float: ...
    @property
    def rtol(self) -> float: ...
