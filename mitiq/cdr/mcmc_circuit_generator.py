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

from typing import (
    Callable,
    Optional,
    Union,
)

import random

import numpy as np
import cirq
from cirq.circuits import Circuit

from mitiq.cdr.random_circuit_generator import RandomCircuitGenerator

from mitiq.interface import (
    class_atomic_one_to_many_converter,
)

from mitiq import Executor, Observable, QPROGRAM, QuantumResult

_GAUSSIAN = "gaussian"
_UNIFORM = "uniform"
_CLOSEST = "closest"


class MCMCCircuitGenerator(RandomCircuitGenerator):
    """Generates new circuits based on a starting circuit using the
    Markov Chain Monte Carlo (MCMC) sampling method."""

    def __init__(
        self,
        fraction_non_clifford: float,
        executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
        observable: Optional[Observable] = None,
        standard_deviation: float = 0.8,
        method_select: str = _UNIFORM,
        method_replace: str = _CLOSEST,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        r"""Initializer for MCMC circuit generator.

        Args:
            fraction_non_clifford: The (approximate) fraction of non-Clifford
                gates in each returned circuit.
            executor: Executes a circuit and returns a `QuantumResult`.
            observable: Observable to compute the expectation value of.
                If None, the `executor` must return an expectation value.
                Otherwise the `QuantumResult` returned by `executor` is used
                to compute the expectation of the observable.
            standard_deviation: The standard deviation of the distribution from
                which circuits are sampled.
            method_select: Method by which non-Clifford gates are selected to
                be replaced by Clifford gates. Options are 'uniform' or
                'gaussian'.
            method_replace: Method by which selected non-Clifford gates are
                replaced by Clifford gates. Options are 'uniform', 'gaussian'
                or 'closest'.
            random_state: Seed for sampling.
        """
        super().__init__(
            fraction_non_clifford,
            method_select=method_select,
            method_replace=method_replace,
            random_state=random_state,
        )
        self.executor = executor
        self.observable = observable
        self.standard_deviation = standard_deviation

    def likelihood(self, x, sigma, mu):
        """Computes the likelihood of a value "x" assuming a normal distribution
        with a given standard deviation and mean."""
        constants = (1 / sigma) * (1 / np.sqrt(2 * np.pi))
        exponentiation = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return constants * exponentiation

    @class_atomic_one_to_many_converter
    def generate_circuits(self, circuit, num_circuits_to_generate):
        """Generates circuits using the MCMC sampling method.

        Args:
            circuit: The starting circuit which is mutated to generate
                new circuits.
            num_circuits_to_generate: The number of circuits to generate.
        """
        if self._random_state is None or isinstance(self._random_state, int):
            self._random_state = np.random.RandomState(self._random_state)

        # Find the non-Clifford operations in the circuit.
        operations = np.array(list(circuit.all_operations()))
        non_clifford_indices_and_ops = np.array(
            [
                [i, op]
                for i, op in enumerate(operations)
                if not cirq.has_stabilizer_effect(op)
            ]
        )

        if len(non_clifford_indices_and_ops) == 0:
            return [circuit] * num_circuits_to_generate

        non_clifford_indices = np.int32(non_clifford_indices_and_ops[:, 0])
        non_clifford_ops = non_clifford_indices_and_ops[:, 1]

        # Replace (some of) the non-Clifford operations.
        near_clifford_circuits = []
        mean_expectation_value = abs(
            self.executor.evaluate([circuit], self.observable)[0],
        )
        last_expectation_value = mean_expectation_value

        while len(near_clifford_circuits) < num_circuits_to_generate:
            new_ops = self._map_to_near_clifford(
                non_clifford_ops,
            )
            operations[non_clifford_indices] = new_ops

            candidate = Circuit(operations)
            candidate_expectation_value = abs(
                self.executor.evaluate([candidate], self.observable)[0],
            )

            # Applying the Metropolis-Hastings rule to determine whether the
            # candidate should be added to the change.

            proposal_likelihood = self.likelihood(
                candidate_expectation_value,
                self.standard_deviation,
                mean_expectation_value,
            )

            last_sample_likelihood = self.likelihood(
                last_expectation_value,
                self.standard_deviation,
                mean_expectation_value,
            )

            if random.random() < min(
                1, proposal_likelihood / last_sample_likelihood
            ):
                near_clifford_circuits.append(candidate)
                last_expectation_value = candidate_expectation_value

        return near_clifford_circuits
