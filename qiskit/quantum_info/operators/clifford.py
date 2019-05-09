# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Clifford operator class.
"""

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.pauli import Pauli


class Clifford(BaseOperator):
    """Clifford table operator class????"""

    def __init__(self, data):
        """Initialize an operator object."""
        if isinstance(data, (QuantumCircuit, Instruction)):
            # If the input is a Terra QuantumCircuit or Instruction we
            # perform a simulation to construct the untiary operator.
            # This will only work if the cirucit or instruction can be
            # defined in terms of unitary gate instructions which have a
            # 'to_matrix' method defined. Any other instructions such as
            # conditional gates, measure, or reset will cause an
            # exception to be raised.
            table = self._init_instruction(data).data
        elif isinstance(data, dict):
            # Initialize from JSON / dict representation
            table = self._from_dict(data).data
        elif isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is clifford table
            data = np.array(data, dtype=np.bool)
            if data.ndim != 2 or data.shape[0] % 2 != 0:
                raise QiskitError("Invalid shape for input Clifford table.")
            if data.shape[0] == data.shape[1]:
                table = data
                table.resize((data.shape[0], data.shape[1] + 1))
            elif data.shape[1] == data.shape[0] + 1:
                table = data
            else:
                raise QiskitError("Invalid shape for input Clifford table.")
        else:
            raise QiskitError("Invalid input data format for Clifford")
        # Determine input and output dimensions
        num_qubits = table.shape[0] // 2
        dims = num_qubits * [2]
        super().__init__('Clifford', table, dims, dims)

    def __repr__(self):
        # Overload repr for Clifford to display more readable JSON form
        return '{}({})'.format(self.rep, self.as_dict())

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        # TODO: Check if a Clifford table is always valid
        return True

    def to_operator(self):
        """Convert operator to matrix operator class"""
        # TODO:
        pass

    def to_instruction(self):
        """Convert to a UnitaryGate instruction."""
        # TODO: circuit decomposition of Clifford
        pass

    def conjugate(self):
        """Return the conjugate of the operator."""
        # TODO
        pass

    def transpose(self):
        """Return the transpose of the operator."""
        # TODO
        pass

    def compose(self, other, qargs=None, front=False):
        """Return the composition channel self∘other.

        Args:
            other (Clifford): an operator object.
            qargs (list): a list of subsystem positions to compose other on.
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            Operator: The composed operator.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        # Check if input is a QuantumCircuit or Instruction and if so
        # use the optimized append function
        if not front and isinstance(other, (QuantumCircuit, Instruction)):
            # Make a copy of the current clifford state
            # TODO: we might need a deep copy here
            op = Clifford(self.data)
            op._append_instruction(other, qargs=qargs)
            return op

        # Otherwise convert to a Clifford and implement compose of the
        # symplectic tables
        if not isinstance(other, Clifford):
            other = Clifford(other)

        # Check dimensions are compatible
        if front and self.input_dims(qargs=qargs) != other.output_dims():
            raise QiskitError(
                'output_dims of other must match subsystem input_dims')
        if not front and self.output_dims(qargs=qargs) != other.input_dims():
            raise QiskitError(
                'input_dims of other must match subsystem output_dims')

        # Full composition of operators
        if qargs is None:
            # TODO
            if front:
                # Composition A(B(input))
                data = None  # TODO: update data
            else:
                # Composition B(A(input))
                data = None  # TODO: update data
            return Clifford(data)
        # Compose with other on subsystem
        return self._compose_subsystem(other, qargs, front)

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (Clifford): a operator subclass object.

        Returns:
            Clifford: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (Clifford): an operator object.

        Returns:
            Clifford: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        return self._tensor_product(other, reverse=True)

    def add(self, other):
        """Not implemented for Cliffords"""
        return NotImplemented

    def subtract(self, other):
        """Not implemented for Cliffords"""
        return NotImplemented

    def multiply(self, other):
        """Not implemented for Cliffords"""
        return NotImplemented

    # ---------------------------------------------------------------------
    # Interface with Pauli object
    # ---------------------------------------------------------------------

    @property
    def num_qubits(self):
        """Return number of qubits for the table."""
        return len(self._input_dims)

    @property
    def table(self):
        """Return the the Clifford table."""
        return self._data[:, 0:-1]

    @table.setter
    def table(self, value):
        """Set Clifford table."""
        self._data[:, 0:-1] = value

    @property
    def phases(self):
        """Return the Clifford phases."""
        return self._data[:, -1]

    @phases.setter
    def phases(self, value):
        """Set Clifford phases."""
        self._data[:, -1] = value

    def stabilizer(self, qubit):
        """Return the qubit stabilizer as a Pauli object"""
        nq = self.num_qubits
        z = self.table[nq + qubit, 0:nq]
        x = self.table[nq + qubit, nq:2 * nq]
        return Pauli(z=z, x=x)

    def set_stabilizer(self, qubit, value):
        """Update the qubit stabilizer row from a Pauli object"""
        index = self.num_qubits + qubit
        if isinstance(value, Pauli):
            # Update from Pauli object
            self.table[index] = np.block([value.z, value.x])
        else:
            # Update table as Numpy array
            self.table[index] = value

    def destabilizer(self, row):
        """Return the destabilizer as a Pauli object"""
        nq = self.num_qubits
        z = self.table[row, 0:nq]
        x = self.table[row, nq:2 * nq]
        return Pauli(z=z, x=x)

    def set_destabilizer(self, qubit, value):
        """Update the qubit destabilizer row from a Pauli object"""
        if isinstance(value, Pauli):
            # Update from Pauli object
            self.table[qubit] = np.block([value.z, value.x])
        else:
            # Update table as Numpy array
            self.table[qubit] = value

    # ---------------------------------------------------------------------
    # JSON / Dict conversion
    # ---------------------------------------------------------------------

    def as_dict(self):
        """Return dictionary (JSON) represenation of Clifford object"""
        # Modify later if we want to include i and -i.
        stabilizers = []
        for qubit in range(self.num_qubits):
            label = self.stabilizer(qubit).to_label()
            phase = self.phases[self.num_qubits + qubit]
            stabilizers.append(('-' if phase else '') + label)
        destabilizers = []
        for qubit in range(self.num_qubits):
            label = self.destabilizer(qubit).to_label()
            phase = self.phases[qubit]
            destabilizers.append(('-' if phase else '') + label)
        return {"stabilizers": stabilizers, "destabilizers": destabilizers}

    @classmethod
    def _from_dict(cls, clifford_dict):
        """Load a Clifford from a dictionary"""

        # Validation
        if not isinstance(clifford_dict, dict) or \
           'stabilizers' not in clifford_dict or \
           'destabilizers' not in clifford_dict:
            raise ValueError("Invalid input Clifford dictionary.")

        stabilizers = clifford_dict['stabilizers']
        destabilizers = clifford_dict['destabilizers']
        if len(stabilizers) != len(destabilizers):
            raise ValueError(
                "Invalid Clifford dict: length of stabilizers and "
                "destabilizers do not match.")
        num_qubits = len(stabilizers)

        # Helper function
        def get_row(label):
            """Return the Pauli object and phase for stabilizer"""
            if label[0] in ['I', 'X', 'Y', 'Z']:
                pauli = Pauli.from_label(label)
                phase = 0
            elif label[0] == '+':
                pauli = Pauli.from_label(label[1:])
                phase = 0
            elif label[0] == '-':
                pauli = Pauli.from_label(label[1:])
                phase = 1
            return pauli, phase

        # Generate identity Clifford on number of qubits
        clifford = cls._init_identity(num_qubits)
        # Update stabilizers
        for qubit, label in enumerate(stabilizers):
            pauli, phase = get_row(label)
            clifford.set_stabilizer(qubit, pauli)
            clifford.phases[num_qubits + qubit] = phase
        # Update destabilizers
        for qubit, label in enumerate(destabilizers):
            pauli, phase = get_row(label)
            clifford.set_destabilizer(qubit, pauli)
            clifford.phases[qubit] = phase
        return clifford

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product operator.

        Args:
            other (Clifford): another Clifford operator.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Clifford: the tensor product operator.

        Raises:
            QiskitError: if other cannot be converted into an Clifford.
        """
        # Convert other to Operator
        if not isinstance(other, Clifford):
            other = Clifford(other)
        # TODO
        pass

    def _compose_subsystem(self, other, qargs, front=False):
        """Return the composition channel."""
        # TODO
        pass

    @classmethod
    def _init_identity(cls, num_qubits):
        """Initialize and identity Clifford table"""
        # Symplectic table
        zeros = np.zeros((num_qubits, num_qubits), dtype=np.bool)
        iden = np.eye(num_qubits, dtype=np.bool)
        phases = np.zeros((num_qubits, 1), dtype=np.bool)
        table = np.block([[zeros, iden, phases], [iden, zeros, phases]])
        return Clifford(table)

    @classmethod
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Instruction to a Clifford."""
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an identity operator of the correct size of the circuit
        op = cls._init_identity(instruction.num_qubits)
        op._append_instruction(instruction)
        return op

    def _index(self):
        """Returns a unique integer index for the Clifford."""
        mat = self.table
        mat = mat.reshape(mat.size)
        ret = int(0)
        for bit in mat:
            ret = (ret << 1) | int(bit)
        mat = self.phases
        mat = mat.reshape(mat.size)
        for bit in mat:
            ret = (ret << 1) | int(bit)
        return ret

    def _append_instruction(self, obj, qargs=None):
        """Update the current Clifford by apply an instruction."""
        # Dictionaries for returning the correct apply function for
        # 1 and 2 qubit clifford table updates.
        cliffords_1q = {
            'id': self._append_id,
            'x': self._append_x,
            'y': self._append_y,
            'z': self._append_z,
            'h': self._append_h,
            's': self._append_s,
            'sdg': self._append_sdg,
            'w': self._append_w,
            'v': self._append_v
        }
        cliffords_2q = {
            'cz': self._append_cz,
            'cx': self._append_cx,
            'swap': self._append_swap
        }
        if not isinstance(obj, Instruction):
            raise QiskitError('Input is not an instruction.')
        if obj.name in cliffords_1q:
            if len(qargs) != 1 or (qargs is None and self.num_qubits != 1):
                raise QiskitError("Incorrect qargs for single-qubit Clifford.")
            if qargs is None:
                qargs = [0]
            # Get the apply function and update operator
            cliffords_1q[obj.name](qargs[0])
        elif obj.name in cliffords_2q:
            if len(qargs) != 2 or (qargs is None and self.num_qubits != 2):
                raise QiskitError("Incorrect qargs for two-qubit Clifford.")
            if qargs is None:
                qargs = [0, 1]
            # Get the apply function and update operator
            cliffords_2q[obj.name](qargs[0], qargs[1])
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError('Invalid Clifford instruction: {}'.format(
                    obj.name))
            for instr, qregs, cregs in obj.definition:
                if cregs:
                    raise QiskitError(
                        'Cannot apply instruction with classical registers: {}'
                        .format(instr.name))
                # Get the integer position of the flat register
                new_qargs = [tup[1] for tup in qregs]
                self._append_instruction(instr, qargs=new_qargs)

    def _append_id(self, qubit):
        """Apply a Pauli "id" gate to a qubit"""
        pass

    def _append_x(self, qubit):
        """Apply a Pauli "x" gate to a qubit"""
        iz = qubit
        self.phases = np.logical_xor(self.phases, self.table[:, iz])

    def _append_y(self, qubit):
        """Apply an Pauli "y" gate to a qubit"""
        iz, ix = qubit, self.num_qubits + qubit
        zx_xor = np.logical_xor(self.table[:, iz], self.table[:, ix])
        self.phases = np.logical_xor(self.phases, zx_xor)

    def _append_z(self, qubit):
        """Apply an Pauli "z" gate to qubit"""
        ix = self.num_qubits + qubit
        self.phases = np.logical_xor(self.phases, self.table[:, ix])

    def _append_h(self, qubit):
        """Apply an Hadamard "h" gate to qubit"""
        iz, ix = qubit, self.num_qubits + qubit
        zx_and = np.logical_and(self.table[:, ix], self.table[:, iz])
        self.phases = np.logical_xor(self.phases, zx_and)
        # Cache X column for qubit
        x_cache = self.table[:, ix].copy()
        # Swap X and Z columns for qubit
        self.table[:, ix] = self.table[:, iz]  # Does this need to be a copy?
        self.table[:, iz] = x_cache

    def _append_s(self, qubit):
        """Apply an phase "s" gate to qubit"""
        iz, ix = qubit, self.num_qubits + qubit
        zx_and = np.logical_and(self.table[:, ix], self.table[:, iz])
        self.phases = np.logical_xor(self.phases, zx_and)
        self.table[:, iz] = np.logical_xor(self.table[:, ix],
                                           self.table[:, iz])

    def _append_sdg(self, qubit):
        """Apply an adjoint phase "sdg" gate to qubit"""
        self._append_z(qubit)
        self._append_s(qubit)

    def _append_v(self, qubit):
        """Apply v gate sd.h"""
        self._append_sdg(qubit)
        self._append_h(qubit)

    def _append_w(self, qubit):
        """Apply w gate v.v"""
        self._append_h(qubit)
        self._append_s(qubit)

    def _append_cx(self, qubit_ctrl, qubit_trgt):
        """Apply a Controlled-NOT "cx" gate"""
        # Helper indices for stabilizer columns
        iz_c, ix_c = qubit_ctrl, self.num_qubits + qubit_ctrl
        iz_t, ix_t = qubit_trgt, self.num_qubits + qubit_trgt
        # Compute phase
        tmp = np.logical_xor(self.table[:, ix_t], self.table[:, iz_c])
        tmp = np.logical_xor(1, tmp)  # Shelly: fixed misprint in logical
        tmp = np.logical_and(self.table[:, iz_t], tmp)
        tmp = np.logical_and(self.table[:, ix_c], tmp)
        self.phases ^= tmp
        # Update stabilizers
        self.table[:, ix_t] = np.logical_xor(self.table[:, ix_t],
                                             self.table[:, ix_c])
        self.table[:, iz_c] = np.logical_xor(self.table[:, iz_t],
                                             self.table[:, iz_c])

    def _append_cz(self, qubit_ctrl, qubit_trgt):
        """Apply a Controlled-z "cx" gate"""
        # TODO: change direct table update if more efficient
        self._append_h(qubit_trgt)
        self._append_cx(qubit_ctrl, qubit_trgt)
        self._append_h(qubit_trgt)

    def _append_swap(self, qubit0, qubit1):
        """Apply SWAP gate between two qubits"""
        # TODO: change direct swap of required rows and cols in table
        self._append_cx(qubit0, qubit1)
        self._append_cx(qubit1, qubit0)
        self._append_cx(qubit0, qubit1)
