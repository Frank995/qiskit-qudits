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

from abc import ABCMeta
from typing import Any

from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction

class _SingletonMeta(ABCMeta):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        *,
        overrides: type[_SingletonInstructionOverrides] | None = ...,
        **kwargs: Any,
    ) -> _SingletonMeta: ...
    def __call__(
        cls,
        *args: Any,
        _force_mutable: bool = ...,
        **kwargs: Any,
    ) -> Any: ...

class _SingletonBase(metaclass=_SingletonMeta):
    __slots__ = ()

    @staticmethod
    def _singleton_lookup_key(*_args: Any, **_kwargs: Any) -> Any: ...

class _SingletonInstructionOverrides(Instruction):
    __slots__ = ()

    @staticmethod
    def _prepare_singleton_instance(
        instruction: Instruction,
    ) -> Instruction: ...
    def copy(self, name: str | None = ...) -> Instruction: ...

class SingletonInstruction(Instruction, _SingletonBase):
    __slots__ = ()
    def __init_subclass__(
        cls,
        *,
        create_default_singleton: bool = ...,
        additional_singletons: tuple[Any, ...] = ...,
        **kwargs: Any,
    ) -> None: ...

class _SingletonGateOverrides(_SingletonInstructionOverrides, Gate):
    __slots__ = ()

class SingletonGate(Gate, _SingletonBase):
    __slots__ = ()
    def __init_subclass__(
        cls,
        *,
        create_default_singleton: bool = ...,
        additional_singletons: tuple[Any, ...] = ...,
        **kwargs: Any,
    ) -> None: ...

class _SingletonControlledGateOverrides(
    _SingletonInstructionOverrides,
    ControlledGate,
):
    __slots__ = ()

class SingletonControlledGate(ControlledGate, _SingletonBase):
    __slots__ = ()
    def __init_subclass__(
        cls,
        *,
        create_default_singleton: bool = ...,
        additional_singletons: tuple[Any, ...] = ...,
        **kwargs: Any,
    ) -> None: ...
