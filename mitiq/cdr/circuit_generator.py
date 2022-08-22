# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Classes corresponding to different zero-noise extrapolation methods."""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import warnings

from mitiq.interface import ( 
    class_atomic_one_to_many_converter,
)

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.polynomial import RankWarning
from scipy.optimize import curve_fit, OptimizeWarning

import cirq
from cirq.circuits import Circuit

from mitiq._typing import QPROGRAM, QuantumResult
from mitiq.observable import Observable
from mitiq.executor import Executor
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.interface import accept_any_qprogram_as_input


class AbstractCircuitGenerator(ABC):
    """Abstract base class that specifies an interface for generating new circuits
    based on a starting circuit. This includes:

        * selecting which gates to swap,
        * generating circuits based on specified gate (or other) constraints,
        * validating the generated circuits.
    """

    def __init__(self) -> None:
        return

    @abstractmethod
    def _swap_operations(
        self,
        op: cirq.ops.Operation,
    ) -> cirq.ops.Operation:
        """Calls the executor function on noise-scaled quantum circuit and
        stores the results.
        """
        raise NotImplementedError

    @abstractmethod
    @class_atomic_one_to_many_converter
    def generate_circuits(
        self,
        circuit: Circuit,
        num_circuit_to_generate: int,
    ) -> List[Circuit]:
        """Calls the function scale_factor_to_expectation_value at each scale
        factor of the factory, and stores the results.

        Args:
            scale_factor_to_expectation_value: A function which inputs a scale
                factor and outputs an expectation value. This does not have to
                involve a quantum processor making this a "classical analogue"
                of the run method.
        """
        raise NotImplementedError
