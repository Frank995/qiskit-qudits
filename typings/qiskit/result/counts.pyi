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

import re
from collections.abc import Iterable, Mapping

class Counts(dict[str, int]):
    bitstring_regex: re.Pattern[str]
    int_raw: dict[int, int] | None
    hex_raw: dict[str, int] | None
    creg_sizes: list[tuple[str, int]] | None
    memory_slots: int | None
    time_taken: float | None

    def __init__(
        self,
        data: Mapping[str | int, int] | Iterable[tuple[str | int, int]],
        time_taken: float | None = ...,
        creg_sizes: list[tuple[str, int]] | None = ...,
        memory_slots: int | None = ...,
    ) -> None: ...
    def most_frequent(self) -> str: ...
    def hex_outcomes(self) -> dict[str, int]: ...
    def int_outcomes(self) -> dict[int, int]: ...
    @staticmethod
    def _remove_space_underscore(bitstring: str) -> int: ...
    def shots(self) -> int: ...
