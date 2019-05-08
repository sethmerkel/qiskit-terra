# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
DensityMatrix quantum state class.
"""

from numbers import Number

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp


class DensityMatrix(QuantumState):
    """DensityMatrix class"""

    def __init__(self, data, dims=None):
        """Initialize a state object."""
        # TODO: conversion form Statevector
        if isinstance(data, (QuantumCircuit, Instruction)):
            # If the input is a Terra QuantumCircuit or Instruction we
            # perform a simulation to construct the output statevector
            # for that circuit assuming that the input is the zero state
            # |0,...,0>.
            # This will only work if the cirucit or instruction can be
            # defined in terms of unitary gate instructions which have a
            # 'to_matrix' method defined. Any other instructions such as
            # conditional gates, measure, or reset will cause an
            # exception to be raised.
            mat = self._init_instruction(data).data
        elif hasattr(data, 'to_operator'):
            # If the data object has a 'to_operator' attribute this is given
            # higher preference than the 'to_matrix' method for initializing
            # an Operator object.
            data = data.to_operator()
            mat = data.data
            if dims is None:
                dims = data.output_dims()
        elif hasattr(data, 'to_matrix'):
            # If no 'to_operator' attribute exists we next look for a
            # 'to_matrix' attribute to a matrix that will be cast into
            # a complex numpy matrix.
            mat = np.array(data.to_matrix(), dtype=complex)
        elif isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is a raw matrix in either a
            # python list or numpy array format.
            mat = np.array(data, dtype=complex)
        else:
            raise QiskitError("Invalid input data format for DensityMatrix")

        # Determine input and output dimensions
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise QiskitError("Invalid DensityMatrix input: not a square matrix.")
        subsystem_dims = self._automatic_dims(dims, mat.shape[0])
        super().__init__('DensityMatrix', mat, subsystem_dims)

    def is_valid(self, atol=None, rtol=None):
        """Return True if trace 1 and positive semidefinite."""
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        # Check trace == 1
        if not np.allclose(self.trace(), 1, rtol=rtol, atol=atol):
            return False
        # Check Hermitian
        if not is_hermitian_matrix(self.data, rtol=rtol, atol=atol):
            return False
        # Check positive semidefinite
        return is_positive_semidefinite_matrix(self.data, rtol=rtol, atol=atol)

    def to_operator(self):
        """Convert to Operator"""
        dims = self.dims()
        return Operator(self.data, input_dims=dims, output_dims=dims)

    def conjugate(self):
        """Return the conjugate of the density matrix."""
        return DensityMatrix(np.conj(self.data), dims=self.dims())

    def trace(self):
        """Return the trace of the density matrix."""
        return np.trace(self.data)

    def purity(self):
        """Return the purity of the quantum state."""
        # For a valid statevector the purity is always 1, however if we simply
        # have an arbitrary vector (not correctly normalized) then the
        # purity is equivalent to the trace squared:
        # P(|psi>) = Tr[|psi><psi|psi><psi|] = |<psi|psi>|^2
        return np.trace(np.dot(self.data, self.data))

    def tensor(self, other):
        """Return the tensor product state self ⊗ other.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        dims = other.dims() + self.dims()
        data = np.kron(self._data, other._data)
        return Operator(data, dims)

    def expand(self, other):
        """Return the tensor product state other ⊗ self.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        dims = self.dims() + other.dims()
        data = np.kron(other._data, self._data)
        return Operator(data, dims)

    def add(self, other):
        """Return the linear combination self + other.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the linear combination self + other.

        Raises:
            QiskitError: if other is not a quantum state, or has
            incompatible dimensions.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        if self.dim != other.dim:
            raise QiskitError("other DensityMatrix has different dimensions.")
        return DensityMatrix(self.data + other.data, self.dims())

    def subtract(self, other):
        """Return the linear operator self - other.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the linear combination self - other.

        Raises:
            QiskitError: if other is not a quantum state, or has
            incompatible dimensions.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        if self.dim != other.dim:
            raise QiskitError("other DensityMatrix has different dimensions.")
        return DensityMatrix(self.data - other.data, self.dims())

    def multiply(self, other):
        """Return the linear operator self + other.

        Args:
            other (complex): a complex number.

        Returns:
            DensityMatrix: the linear combination other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return DensityMatrix(other * self.data, self.dims())

    def evolve(self, other, qargs=None):
        """Evolve a quantum state by an operator.

        Args:
            other (Operator or QuantumChannel
                   or Instruction or Circuit): The operator to evolve by.
            qargs (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            QuantumState: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
            specified QuantumState subsystem dimensions.
        """
        # Evolution by a circuit or instruction
        if isinstance(other, (QuantumCircuit, Instruction)):
            return self._evolve_instruction(other, qargs=qargs)
        # Evolution by a QuantumChannel
        if isinstance(other, QuantumChannel) or hasattr(other, 'to_quantumchannel'):
            return self._evolve_superop(SuperOp(other), qargs=qargs)
        # Unitary evolution by an Operator
        return self._evolve_operator(Operator(other), qargs=qargs)

    @property
    def _shape(self):
        """Return the tensor shape of the matrix operator"""
        return 2 * tuple(reversed(self.dims()))

    def _evolve_operator(self, other, qargs=None):
        """Return a new DensityMatrix by applying an Operator."""
        if qargs is None:
            # Evolution on full matrix
            if self._dim != other._input_dim:
                raise QiskitError(
                    "Operator input dimension is not equal to density matrix dimension."
                )
            mat = np.dot(other.data, self.data).dot(other.adjoint().data)
            return DensityMatrix(mat, dims=other.output_dims())
        # Otherwise we are applying an operator only to subsystems
        # Check dimensions of subsystems match the operator
        if self.dims(qargs) != other.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        # Reshape statevector and operator
        tensor = np.reshape(self.data, self._shape)
        # Construct list of tensor indices of statevector to be contracted
        num_indices = len(self.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs]
        # Left multiple by mat
        mat = np.reshape(other.data, other._shape)
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        # Right multiply by mat ** dagger
        adj = other.adjoint()
        mat_adj = np.reshape(adj.data, adj._shape)
        tensor = Operator._einsum_matmul(tensor, mat_adj, indices, num_indices, True)
        # Replace evolved dimensions
        new_dims = list(self.dims())
        for i, qubit in enumerate(qargs):
            new_dims[qubit] = other._output_dims[i]
        new_dim = np.product(new_dims)
        return DensityMatrix(np.reshape(tensor, (new_dim, new_dim)), dims=new_dims)

    def _evolve_superop(self, other, qargs=None):
        """Return a new DensityMatrix by applying a SuperOp."""
        if qargs is None:
            # Evolution on full matrix
            if self._dim != other._input_dim:
                raise QiskitError(
                    "Operator input dimension is not equal to density matrix dimension."
                )
            # We reshape in column-major vectorization (Fortran order in Numpy)
            # since that is how the SuperOp is defined
            vec = np.ravel(self.data, order='F')
            mat = np.reshape(np.dot(other.data, vec), (self.dim, self.dim), order='F')
            return DensityMatrix(mat, dims=other.output_dims())
        # Otherwise we are applying an operator only to subsystems
        # Check dimensions of subsystems match the operator
        if self.dims(qargs) != other.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        # Reshape statevector and operator
        tensor = np.reshape(self.data, self._shape, order='F')
        mat = np.reshape(other.data, other._shape)
        # Construct list of tensor indices of statevector to be contracted
        num_indices = len(self.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs
                   ] + [2 * num_indices - 1 - qubit for qubit in qargs]
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        # Replace evolved dimensions
        new_dims = list(self.dims())
        for i, qubit in enumerate(qargs):
            new_dims[qubit] = other._output_dims[i]
        new_dim = np.product(new_dims)
        # reshape tensor to density matrix
        tensor = np.reshape(tensor, (new_dim, new_dim), order='F')
        return DensityMatrix(tensor, dims=new_dims)

    def _append_instruction(self, other, qargs=None):
        """Update the current Statevector by applying an instruction."""
        # Try evolving by a matrix operator (unitary-like evolution)
        mat = Operator._instruction_to_matrix(other)
        if mat is not None:
            self._data = self._evolve_operator(Operator(mat), qargs=qargs).data
            return
        # Otherwise try evolving by a Superoperator
        chan = SuperOp._instruction_to_superop(other)
        if chan is not None:
            # Evolve current state by the superoperator
            self._data = self._evolve_superop(chan, qargs=qargs).data
            return
        # If the instruction doesn't have a matrix defined we use its
        # circuit decomposition definition if it exists, otherwise we
        # cannot compose this gate and raise an error.
        if other.definition is None:
            raise QiskitError('Cannot apply Instruction: {}'.format(
                other.name))
        for instr, qregs, cregs in other.definition:
            if cregs:
                raise QiskitError(
                    'Cannot apply instruction with classical registers: {}'
                    .format(instr.name))
            # Get the integer position of the flat register
            new_qargs = [tup[1] for tup in qregs]
            self._append_instruction(instr, qargs=new_qargs)

    def _evolve_instruction(self, obj, qargs=None):
        """Return a new statevector by applying an instruction."""
        if isinstance(obj, QuantumCircuit):
            obj = obj.to_instruction()
        vec = DensityMatrix(self.data, dims=self.dims())
        vec._append_instruction(obj, qargs=qargs)
        return vec

    @classmethod
    def _init_instruction(cls, instruction):
        """Initialize from output of an instruction applied to zero state."""
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an the statevector in the all |0> state
        n_qubits = instruction.num_qubits
        init = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
        init[0, 0] = 1
        vec = DensityMatrix(init, dims=n_qubits * [2])
        vec._append_instruction(instruction)
        return vec
