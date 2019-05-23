# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Pauli operator class.
"""
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.instruction import Instruction
from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import Operator


class PauliOp(BaseOperator):
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
            pvector = self._init_instruction(data).data
        elif isinstance(data, dict):
            # Initialize from JSON / dict representation
            pvector = self._from_dict(data).data
        elif isinstance(data, str):
            # Initialize from a pauli string
            pvector = self._from_label(data).data
        elif hasattr(data, 'to_operator'):
            # If the data object has a 'to_operator' attribute this is given
            # higher preference than the 'to_matrix' method for initializing
            # an Operator object.
            data = data.to_operator()
            pvector = self._from_matrix(data.data).data
        elif hasattr(data, 'to_matrix'):
            # If no 'to_operator' attribute exists we next look for a
            # 'to_matrix' attribute to a matrix that will be cast into
            # a complex numpy matrix.
            pvector =self._from_matrix(np.array(data.to_matrix(), dtype=complex)).data
        elif isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is a stabilzer [z , x] or unitary matrix
            data = np.array(data)
            if data.ndim == 1  :
                if 2*(len(data)//2) != len(data):
                    raise QiskitError("Invalid shape for input Pauli stabilizer.")
                pvector = np.array(data, dtype=np.bool)     
            elif data.ndim == 2  :
                pvector =self._from_matrix(np.array(data, dtype=complex)).data
            else:
                raise QiskitError("Invalid shape for input Pauli table.")   
        else:
            raise QiskitError("Invalid input data format for Pauli")
        # Determine input and output dimensions
        num_qubits = len(pvector)//2
        dims = num_qubits * [2]
        super().__init__('PauliOp', pvector, dims, dims)

    def __repr__(self):
        # Overload repr for Pauli to display more readable JSON form
        return '{}({})'.format(self.rep, self.as_dict())

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        # Pauli table is always valid
        return True

    def to_operator(self):
        """Convert operator to matrix operator class"""
        return Operator(self.to_instruction())

    def to_instruction(self):
        """Convert to a UnitaryGate instruction."""
        from qiskit.extensions.standard import IdGate, XGate, YGate, ZGate
        gates = {'I': IdGate(), 'X': XGate(), 'Y': YGate(), 'Z': ZGate()}
        label = str(self)
        n_qubits = self.num_qubits
        qreg = QuantumRegister(n_qubits)
        circuit = QuantumCircuit(qreg, name='Pauli:{}'.format(label))
        for i, pauli in enumerate(reversed(label)):
            circuit.append(gates[pauli], [qreg[i]])
        return circuit.to_instruction()
       
    def conjugate(self):
        """Return the conjugate of the operator."""
        return self

    def transpose(self):
        """Return the transpose of the operator."""
        return self

    def compose(self, other, qargs=None, front=False):
        """Return the composition channel self∘other.

        Args:
            other (Paali): an operator object.
            qargs (list): a list of subsystem positions to compose other on.
            front (bool): Pauli's commute, so front back is irrelevant

        Returns:
            Operator: The composed operator.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        
        # Convert to Operator
        if not isinstance(other, PauliOp):
            other = PauliOp(other)
        # Check dimensions are compatible
        if front and self.input_dims(qargs=qargs) != other.output_dims():
            raise QiskitError(
                'output_dims of other must match subsystem input_dims')
        if not front and self.output_dims(qargs=qargs) != other.input_dims():
            raise QiskitError(
                'input_dims of other must match subsystem output_dims')
        # Full composition of operators
        if qargs is None:
            output=PauliOp(str(self))
            output.data = np.logical_xor(output.data,other.data)
            return output
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
    
    def __str__(self):
        """Output the Pauli label."""
        label = ''
        nq = self.num_qubits
        for z, x in zip(reversed(self._data[:nq]),reversed(self._data[nq:])):
            if not z and not x:
                label = ''.join([label, 'I'])
            elif not z and x:
                label = ''.join([label, 'X'])
            elif z and not x:
                label = ''.join([label, 'Z'])
            else:
                label = ''.join([label, 'Y'])
        return label
    
    # ---------------------------------------------------------------------
    # Interface ???
    # ---------------------------------------------------------------------

    @property
    def num_qubits(self):
        """Return number of qubits for the table."""
        return len(self._input_dims)

    def to_label(self):
        """Present the pauli labels in I, X, Y, Z format.

        Order is $q_{n-1} .... q_0$

        Returns:
            str: pauli label
        """
        return str(self)

    
    # ---------------------------------------------------------------------
    # JSON / Dict conversion
    # ---------------------------------------------------------------------

    def as_dict(self):
        """Return dictionary (JSON) represenation of Clifford object"""
        # Modify later if we want to include i and -i.
        return {"paulistring": str(self)}

    @classmethod
    def _from_dict(cls, pauli_dict):
        """Load a Pauli from a dictionary"""

        # Validation
        if not isinstance(pauli_dict, dict) or \
           'paulistring' not in clifford_dicts:
            raise ValueError("Invalid input Pauli dictionary.")

        paulistring = pauli_dict['paulistring']
        
        return cls._from_label(paulistring).data
    
    def _from_label(cls, label):
        """Load a Pauli from a string"""
        nq = len(label)
        pvector = np.zeros(2*nq,dtype=bool)
        for i, char in enumerate(label):
            if char == 'X':
                pvector[-i -1] = True
            elif char == 'Z':
                pvector[-nq -i -1] = True
            elif char == 'Y':
                pvector[-nq -i - 1] = True
                pvector[-i - 1] = True
            elif char != 'I':
                raise QiskitError("Pauli string must be only consisted of 'I', 'X', "
                                  "'Y' or 'Z' but you have {}.".format(char))
        return PauliOp(pvector)
    
    def _from_matrix(cls, mat):
        """Identify a Pauli from a matrix"""
        mat=np.array(mat,dtype=complex)
        if mat.ndim != 2 or  mat.shape[0] != mat.shape[1] or int(np.ceil(np.log2(mat.shape[0]))) != int(np.floor(np.log2(mat.shape[0]))):
            raise QiskitError("Pauli matrix has wrong shape, must be 2^n x 2^n ")
        nq = int(np.ceil(np.log2(mat.shape[0])))
        PauliOps = pauliop_group(nq)
        for Op in PauliOps:
            if np.abs(np.trace(np.dot(mat,Op.to_operator().data )))**2/4.**nq > 1-cls._atol:
                return Op
        
        raise QiskitError("Matrix doesn't appear to be a Pauli matrix")
    
    def _tensor_product(self, other, reverse=False):
        """Return the tensor product operator.

        Args:
            other (Operator): another operator.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Operator: the tensor product operator.

        Raises:
            QiskitError: if other cannot be converted into an Operator.
        """
        # Convert other to Operator
        if not isinstance(other, PauliOp):
            other = PauliOp(other)
        nq = self.num_qubits
        nqother = other.num_qubits
        if reverse:
            data = str(self)+str(other)
        else:
            data = str(other)+str(self)
        return PauliOp(data)


    def _compose_subsystem(self, other, qargs, front=False):
        """Return the composition channel."""
        # TODO
        pass

    @classmethod
    def _init_identity(cls, num_qubits):
        """Initialize and identity Pauli """
        # Identity Pauli
        pvector = np.zeros(2*num_qubits,dtype=bool)
        return PauliOp(pvector)

    @classmethod
    def _compose_subsystem(self, other, qargs, front=False):
        """Return the composition channel."""
        # Compute tensor contraction indices from qargs (no phase so front it irrelevant)
        nq = self.num_qubits
        nqother = other.num_qubits
        output=PauliOp(str(self))
        for ind, qubit in enumerate(qargs):
            output.data[-1-qubit] = output.data[-1-qubit]^other.data[-1-qubit]
            output.data[-nq-1-qubit] = output.data[-nq-1-qubit]^other.data[-nqother-1-qubit]
        return output

    
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Instruction to a Pauli."""
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an identity operator of the correct size of the circuit
        op = cls._init_identity(instruction.num_qubits)
        op._append_instruction(instruction)
        return op


    def _append_instruction(self, obj, qargs=None):
        """Update the current Clifford by apply an instruction."""
        # Dictionaries for returning the correct apply function for
        # 1 and 2 qubit clifford table updates.
        paulis_1q = {
            'id': self._append_id,
            'x': self._append_x,
            'y': self._append_y,
            'z': self._append_z   
        }

        if not isinstance(obj, Instruction):
            raise QiskitError('Input is not an instruction.')
        if obj.name in paulis_1q:
            if len(qargs) != 1 or (qargs is None and self.num_qubits != 1):
                raise QiskitError("Incorrect qargs for single-qubit Pauli.")
            if qargs is None:
                qargs = [0]
            # Get the apply function and update operator
            paulis_1q[obj.name](qargs[0])
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError('Invalid Pauli instruction: {}'.format(
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
        self.data[self.num_qubits+qubit] = not self.data[self.num_qubits+qubit]
        

    def _append_y(self, qubit):
        """Apply an Pauli "y" gate to a qubit"""
        self.data[qubit] = not self.data[qubit]
        self.data[self.num_qubits+qubit] = not self.data[self.num_qubits+qubit]

    def _append_z(self, qubit):
        """Apply an Pauli "z" gate to qubit"""
        self.data[qubit] = not self.data[qubit] 
        

def pauliop_group(number_of_qubits, case='weight'):
    """Return the Pauli group with 4^n elements.

    The phases have been removed.
    case 'weight' is ordered by Pauli weights and
    case 'tensor' is ordered by I,X,Y,Z counting lowest qubit fastest.

    Args:
        number_of_qubits (int): number of qubits
        case (str): determines ordering of group elements ('weight' or 'tensor')

    Returns:
        list: list of Pauli objects

    Raises:
        QiskitError: case is not 'weight' or 'tensor'
        QiskitError: number_of_qubits is larger than 4
    """
    if number_of_qubits < 5:
        temp_set = []

        if case == 'weight':
            tmp = pauliop_group(number_of_qubits, case='tensor')
            # sort on the weight of the Pauli operator
            return sorted(tmp, key=lambda x: -np.count_nonzero(
                np.array(x.to_label(), 'c') == b'I'))
        elif case == 'tensor':
            # the Pauli set is in tensor order II IX IY IZ XI ...
            for k in range(4 ** number_of_qubits):
                z = np.zeros(number_of_qubits, dtype=np.bool)
                x = np.zeros(number_of_qubits, dtype=np.bool)
                # looping over all the qubits
                for j in range(number_of_qubits):
                    # making the Pauli for each j fill it in from the
                    # end first
                    element = (k // (4 ** j)) % 4
                    if element == 1:
                        x[j] = True
                    elif element == 2:
                        z[j] = True
                        x[j] = True
                    elif element == 3:
                        z[j] = True
                current = []
                current.extend(z)
                current.extend(x)
                temp_set.append(PauliOp(current))
            return temp_set
        else:
            raise QiskitError("Only support 'weight' or 'tensor' cases "
                              "but you have {}.".format(case))

    raise QiskitError("Only support number of qubits is less than 5")    
