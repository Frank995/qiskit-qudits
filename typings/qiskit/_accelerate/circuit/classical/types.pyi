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

from enum import Enum

class Type: ...
class Bool(Type): ...

class Uint(Type):
    def __init__(self, width: int) -> None: ...

class Float(Type): ...
class Duration(Type): ...

class Ordering(Enum):
    LESS = ...
    NONE = ...

class CastKind(Enum):
    EQUAL = 1
    IMPLICIT = 2
    LOSSLESS = 3
    DANGEROUS = 4
