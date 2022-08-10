# Copyright (C) 2021 Unitary Fund
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

"""Functions for mapping circuits to (near) Clifford circuits."""
from typing import List, Any

import numpy as np

import cirq
from cirq.circuits import Circuit

from mitiq.interface import accept_any_qprogram_as_input

# Z gates with these angles/exponents are Clifford gates.
_CLIFFORD_EXPONENTS = np.array([0.0, 0.5, 1.0, 1.5])
_CLIFFORD_ANGLES = [exponent * np.pi for exponent in _CLIFFORD_EXPONENTS]


@accept_any_qprogram_as_input
def is_clifford(circuit: Circuit) -> bool:
    """Returns True if the input argument is Clifford, else False.

    Args:
        circuit: A single operation, list of operations, or circuit.
    """
    return all(
        cirq.has_stabilizer_effect(op) for op in circuit.all_operations()
    )


@accept_any_qprogram_as_input
def count_non_cliffords(circuit: Circuit) -> int:
    """Returns the number of non-Clifford operations in the circuit. Assumes
    the circuit consists of only Rz, Rx, and CNOT operations.

    Args:
        circuit: Circuit to count the number of non-Clifford operations in.
    """
    return sum(
        not cirq.has_stabilizer_effect(op) for op in circuit.all_operations()
    )


def random_clifford(
    num_angles: int, random_state: np.random.RandomState
) -> np.ndarray:
    """Returns an array of Clifford angles chosen uniformly at random.

    Args:
        num_angles: Number of Clifford angles to return in array.
        random_state: Random state for sampling.
    """
    return np.array(
        [random_state.choice(_CLIFFORD_ANGLES) for _ in range(num_angles)]
    )


@np.vectorize
def closest_clifford(angles: np.ndarray) -> float:
    """Returns the nearest Clifford angles to the input angles.

    Args:
        non_Clifford_ops: Non-Clifford opperations.
    """
    ang_scaled = angles / (np.pi / 2)
    # if just one min value, return the corresponding nearest cliff.
    if (
        abs((ang_scaled / 0.5) - 1) > 10 ** (-6)
        and abs((ang_scaled / 0.5) - 3) > 10 ** (-6)
        and (abs((ang_scaled / 0.5) - 5) > 10 ** (-6))
    ):
        index = int(np.round(ang_scaled)) % 4
        return _CLIFFORD_ANGLES[index]
    # If equidistant between two Clifford angles, randomly choose one.
    else:
        index_list = [ang_scaled - 0.5, ang_scaled + 0.5]
        index = int(np.random.choice(index_list))
        return _CLIFFORD_ANGLES[index]


@np.vectorize
def is_clifford_angle(
    angles: np.ndarray,
    tol: float = 10**-5,
) -> bool:
    """Function to check if a given angle is Clifford.

    Args:
        angles: rotation angle in the Rz gate.
    """
    angles = angles % (2 * np.pi)
    closest_clifford_angle = closest_clifford(angles)
    if abs(closest_clifford_angle - angles) < tol:
        return True
    return False


def angle_to_proximities(angle: np.ndarray, sigma: float) -> List[float]:
    """Returns probability distribution based on distance from angles to
    Clifford gates.

    Args:
        angle: angle to form probability distribution.

    Returns:
        discrete value of probability distribution calculated from
        exp(-(diff/sigma)^2) where diff is the distance from each angle and the
        Clifford gates.
    """
    s_matrix = cirq.unitary(cirq.S)
    rz_matrix = cirq.unitary(cirq.rz(angle % (2 * np.pi)))
    # TODO: Update loop / if.
    dists = []
    for exponent in range(4):
        if exponent == 0:
            exponent = 4
        diff = np.linalg.norm(rz_matrix - s_matrix**exponent)
        dists.append(np.exp(-((diff / sigma) ** 2)))
    return dists


@np.vectorize
def angle_to_proximity(angle: np.ndarray, sigma: float) -> float:
    """Returns probability distribution based on distance from angles to
    Clifford gates.

    Args:
        angle: angle to form probability distribution.

    Returns:
        discrete value of probability distribution calculated from
        exp(-(dist/sigma)^2) where dist = sum(dists) is the
        sum of distances from each Clifford gate.
    """
    dists = angle_to_proximities(angle, sigma)
    return np.max(dists)


@np.vectorize
def probabilistic_angle_to_clifford(
    angles: np.ndarray,
    sigma: float,
    random_state: np.random.RandomState,
) -> float:
    """Returns a Clifford angle sampled from the distribution

                        prob = exp(-(dist/sigma)^2)

    where dist is the Frobenius norm from the 4 clifford angles and the gate
    of interest.

    Args:
        angles: Non-Clifford angles.
        sigma: Width of probability distribution.
    """

    dists = angle_to_proximities(angles, sigma)

    cliff_ang = random_state.choice(
        _CLIFFORD_ANGLES, 1, replace=False, p=np.array(dists) / np.sum(dists)
    )
    return cliff_ang
