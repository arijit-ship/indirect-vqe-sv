"""
The Hamiltonian definations.
"""

from typing import List

from openfermion.ops.operators.qubit_operator import QubitOperator
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text

def create_xy_hamiltonian(num_qubits: int) -> Observable:
    r"""
    Creates a one-dimensional XY-model Hamiltonian.

    Mathematical form:

        .. math::
        H_\text{XY} = \sum_{k=1}^{N-1}X_k X_{k+1} + Z_k Z_{k+1}

    Args:
        num_qubits (int): The number of qubits.

    Returns:
        Observable: Qulacs observable representing the Hamiltonian.
    """
    hami = QubitOperator()

    for i in range(num_qubits - 1):
        hami += QubitOperator(f"X{i} X{i+1}")
        hami += QubitOperator(f"Z{i} Z{i+1}")

    return create_observable_from_openfermion_text(str(hami))

def create_tf_ising_hamiltonian(num_qubits: int, j: float, h: float) -> Observable:

    r"""
    Creates a one-dimensional transverse-field Ising Hamiltonian.

    Mathematical form:

        .. math::
        H_\text{TFI} = \sum_{k=1}^{N-1}J_k X_k X_{k+1} + \sum_{k=0}^{N} h_k Z_k

    Args:
        num_qubits (int): The number of qubits.
        j (float): Coupling coefficient with values between 0.0 and 1.0.
        h (float): Magnetic field with values between 0.0 and 1.0.

    Returns:
        Observable: Qulacs observable representing the Hamiltonian.
    """
    hami = QubitOperator()

    for i in range(num_qubits - 1):
        hami += j * QubitOperator(f"X{i} X{i+1}")

    for i in range(num_qubits):
        hami += h * QubitOperator(f"Z{i}")

    return create_observable_from_openfermion_text(str(hami))
