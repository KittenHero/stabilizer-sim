'''
Based on the Gottesman-Knill Theorem on simulating
stabilizer circuits in O(m n^2) time and O(n^2) space
where:
    m is the number of operations in the clifford group
    n is the number of qubits

This simulation uses sparse encoding for simplicity,
as more efficient solutions are available elsewhere.
Original: scottaaronson.com/chp

[0] Nielsen and Chuagn, Quantum Computing and Quantum Information, Chapter 10
[1] Aaronson and Gottesman, Improved Simulation of Stabilizer Circuit, arxiv:quant-ph/0406196
'''

__all__ = [
    'QState', 'Measure',
    'QCircuit', 'ControlledNot', 'Hadamard', 'Phase',
    'PauliX', 'PauliZ', 'ControlledZ', 'Swap'
]

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple, MutableSequence
from itertools import chain
import re
import random

class QState:
    '''QState stores generators for the stabilizers and destabilizers of the state
    The stabilizers of a state |ψ⟩ is defined as the set S = { g | g|ψ⟩ = |ψ⟩ }
    A set of n independent Pauli operators g ∈ {1,i,-1,-i}*{I,X,Y,Z}^n
    uniquely determines a state of n qbits in a stabilizer circuit
    Pauli operators acting on a qbit has the following effect:
        X|ψ⟩ = |0⟩⟨1|ψ⟩ + |1⟩⟨0|ψ⟩ (bit flip)
        Z|ψ⟩ = |0⟩⟨0|ψ⟩ - |1⟩⟨1|ψ⟩ (phase flip)
        I|ψ⟩ = |ψ⟩
        Y|ψ⟩ = i Z X |ψ⟩
    '''
    I = 0b00
    Z = 0b01
    X = 0b10
    Y = 0b11

    __slots__ = ('s_gen', 's_phase', 'd_gen', 'd_phase')
    
    def __init__(self, s_gen, s_phase, d_gen, d_phase):
        '''
        Initialize a state with the specified tableau configuration

        Parameters
        ----------
        s_gen : 
            The generator matrix of the Pauli stabilizer
        s_phase :
            The phase of the corresponding generator in s_gen
        d_gen :
            The corresponding generator matrix of the destabilizer
        s_phase :
            The phase of the corresponding generator in d_gen

        Raises
        ------
        AssertionError
            The corresponding rows of d_gen and s_gen must anti-commute
            while commuting with the rest of the generators
            Each generator must also be independent
        '''
        assert len(s_gen) == len(d_gen) == len(s_phase) == len(d_phase)

        n = len(s_gen)
        def commute(g, h):
            return sum(self.levi_civita_mod4(gi, hi, gi ^ hi) for gi, hi in zip(g, h)) % 2 == 0

        # corresponding pairs don't commute O(n)
        for pi, qi in zip(s_gen, d_gen):
            assert not commute(pi, qi)
        # stabilizers self commute O(n^2)
        for i, pi in enumerate(s_gen):
            for pj in s_gen[i + 1:]:
                assert commute(pi, pj)
        # destabilizers self commute O(n^2)
        for i, qi in enumerate(d_gen):
            for qj in d_gen[i + 1:]:
                assert commute(qi, qj)
        # stabilizers commute with destabilizers O(n^2)
        for i, pi in enumerate(s_gen):
            for j, qj in enumerate(d_gen):
                if i == j: continue
                assert commute(pi, qj)
        # stabilizers and destabilizers produce the full paulis
        paulis = { self.X, self.Y, self.Z }
        for g_k in zip(*s_gen, *d_gen):
            assert len(set(g_k) & paulis) > 1

        self.s_gen = s_gen
        self.s_phase = s_phase
        self.d_gen = d_gen
        self.d_phase = d_phase

    @classmethod
    def zeros(cls, size):
        '''Initialize the state n qbits set to zero'''
        s_gen = [
            [cls.I if r != c else cls.Z for c in range(size)]
            for r in range(size)
        ]
        d_gen = [
            [cls.I if r != c else cls.X for c in range(size)]
            for r in range(size)
        ]
        s_phase = [0] * size
        d_phase = [0] * size
        return cls(s_gen, s_phase, d_gen, d_phase)

    def apply_clifford(self, qgate):
        '''Apply a clifford gate on the state'''

        for i, (s, d) in enumerate(zip(self.s_gen, self.d_gen)):
            self.s_phase[i] ^= qgate.transform_generator(s)
            self.d_phase[i] ^= qgate.transform_generator(d)

        return self

    def gaussian(self):
        '''Perform gaussian elimination
        to get a minimal set of X and Y in the upper triangular form
        Useful for printing basis state
        
        Returns
        -------
            Number of X in the stabilizer
        '''
        n = len(self)
        def min_r(target, col, start):
            for i, g in enumerate(self.s_gen[start:], start):
                if g[col] & target: return i
            else: return None
        def gaussian_help(target, start_row=0):
            for piv_c in range(n):
                target_r = min_r(target, piv_c, start_row)
                if target_r is None: continue

                self.swap_row(target_r, start_row)

                piv_sg, piv_dg = self.s_gen[start_row], self.d_gen[start_row]
                s_phase, d_phase = self.s_phase[start_row], self.d_phase[start_row]
                for i in range(start_row + 1, n):
                    if not self.s_gen[i][piv_c] & target: continue
                    self.s_phase[i] ^= s_phase ^ self.row_mul(self.s_gen[i], piv_sg)
                    d_phase ^= self.d_phase[i] ^ self.row_mul(piv_dg, self.d_gen[i])
                self.d_phase[start_row] = d_phase
                start_row += 1
            return start_row
        x_count = gaussian_help(self.X)
        gaussian_help(self.Z, x_count)
        return x_count

    def swap_row(self, i, j):
        self.s_phase[i], self.s_phase[j] = self.s_phase[j], self.s_phase[i]
        self.d_phase[i], self.d_phase[j] = self.d_phase[j], self.d_phase[i]
        self.s_gen[i], self.s_gen[j] = self.s_gen[j], self.s_gen[i]
        self.d_gen[i], self.d_gen[j] = self.d_gen[j], self.d_gen[i]

    def __len__(self):
        '''The number of qbits being simulated'''
        return len(self.s_phase)

    @classmethod
    def row_mul(cls, ri, rj):
        '''Multiply the second generator into the first in-place and return the phase'''
        phase = 0
        for k, (ik, jk) in enumerate(zip(ri, rj)):
            prod = ik ^ jk
            phase = (phase + cls.levi_civita_mod4(ik, jk, prod)) & 0b11
            ri[k] = prod
        return phase >> 1

    @classmethod
    def levi_civita_mod4(cls, *order):
        I,Z,X,Y = cls.I, cls.Z, cls.X, cls.Y
        if I in order: return 0
        if order in ((Z,X,Y), (X,Y,Z), (Y,Z,X)): return 1
        else: return 3

    def ket(self, max_ket=10):
        '''The string representation of eigenstate in ket notation'''
        x_count = self.gaussian()
        if x_count > max_ket: raise ValueError('State is too big to be written')
        def gen_basis(gen, phase):
            phase = (sum(p == self.Y for p in gen) + 2*phase) & 0b11
            pre = {0: '+', 1: '+i', 2: '-', 3: '-i'}[phase]
            bits = ''.join('1' if g & self.X else '0' for g in gen)
            return f'{pre}|{bits}⟩'
        # behold magic
        n = len(self)
        gen = [self.I] * n
        for i in reversed(range(x_count, n)):
            phase, g = self.s_phase[i], self.s_gen[i]
            for j in reversed(range(n)):
                if not g[j] & self.Z: continue
                k = j
                if gen[j] & self.X: phase ^= 1
            if phase: gen[k] ^= self.X
        phase = 0
        bases = [gen_basis(gen, phase)]
        for i in range((1 << x_count) - 1):
            wot = i ^ (i + 1)
            for j in range(x_count):
                if not wot & (1 << j): continue
                phase ^= self.s_phase[j] ^ self.row_mul(gen, self.s_gen[j])
            bases.append(gen_basis(gen, phase))
        return '\n'.join(bases)


    def __str__(self):
        '''matrix representation of the destabilizer and stabilizer'''
        pauli_str = { self.I: 'I', self.X: 'X', self.Y: 'Y', self.Z:'Z' }
        phase_str = { 0: '+', 1: '-' }
        d_str = []
        s_str = []
        sep = '{}' + '-' * len(self.s_phase) + '\n'
        for s_phase, s_generator, d_phase, d_generator in zip(self.s_phase, self.s_gen, self.d_phase, self.d_gen):
            d_str.append(phase_str[d_phase])
            d_str.extend(map(pauli_str.get, d_generator))
            d_str.append('\n')

            s_str.append(phase_str[s_phase])
            s_str.extend(map(pauli_str.get, s_generator))
            s_str.append('\n')

        return sep.format('D') + ''.join(d_str) + sep.format('S') + ''.join(s_str).strip()

    def __repr__(self):
        return f'''{self.__class__.__name__}(
            {self.s_gen},
            {self.s_phase},
            {self.d_gen},
            {self.d_phase}
        )'''


@dataclass
class Measure:
    '''For measuring a qbit in the computational, i.e. Z, basis
         Z     -> 0
        -Z     -> 1
    {+/-}{X,Y} -> ?
    ''' 
    target: int
    outcome: Union[int, None] = None

    def __call__(self, qstate):
        n = self.target
        clifford = [QState.I] * len(qstate)

        if all(not g[n] & QState.X for g in qstate.s_gen):
            phase = 0
            for sp, sg, dg in zip(qstate.s_phase, qstate.s_gen, qstate.d_gen):
                if not dg[n] & QState.X: continue
                phase ^= sp ^ QState.row_mul(clifford, sg)
            self.outcome = phase
        else:
            phase = 0
            for i, (p, g) in enumerate(chain(zip(qstate.d_phase, qstate.d_gen),
                                             zip(qstate.s_phase, qstate.s_gen))):
                if not g[n] & QState.X: continue
                phase ^=  p ^ QState.row_mul(clifford, g)
                k = i - len(qstate)
            qstate.d_gen[k] = clifford
            qstate.s_gen[k] = [QState.I] * len(qstate)
            qstate.s_gen[k][n] = QState.Z
            qstate.d_phase[k] = phase
            qstate.s_phase[k] = self.outcome = random.randint(0, 1)
        return self.outcome

    def inverse(self, qstate):
        raise NotImplemented('Impossible')


#####################################

class CliffordGate(metaclass=ABCMeta):
    '''CliffordGate acts uniformly on the generators of QState
    A quantum gate U acting on a state |ψ⟩ stabilised by a generator g ∈ S
    produces a state U|ψ⟩ = Ug|ψ⟩ = UgU'U|ψ⟩ stabilized by UgU'
    '''
    __slots__ = ()

    def __call__(self, qstate):
        '''Apply this gate to the given state'''
        return qstate.apply_clifford(self)
    
    @abstractmethod
    def transform_generator(self, generator):
        '''Apply the gate to the generator in-place and return the phase'''
        pass

    @abstractmethod
    def inverse(self, qstate):
        '''Apply the inverse of this gate to the given state'''
        pass


@dataclass
class ControlledNot(CliffordGate):
    '''
    ----O----
        |
       _|_
      |   |
    --| X |--
      |___|

    cX {II,ZI,IX,ZX} cX ->  {II,ZI,IX,ZX}
    cX    {IZ,IY}    cX ->     {ZZ,ZY}
    cX    {XI,YI}    cX ->     {XX,YX}
    cX      XY       cX ->        YZ
    cX      XZ       cX ->       -YY
    '''
    target: Tuple[int, int]
    __slots__ = ('target',)

    def transform_generator(self, g):
        PHASE_CHANGE = ((QState.X, QState.Z), (QState.Y, QState.Y))
        a, b = self.target
        ga, gb = g[a], g[b]

        g[a] ^= (gb & QState.Z)
        g[b] ^= (ga & QState.X)

        return 1 if (ga, gb) in PHASE_CHANGE else 0

    def inverse(self, qstate):
        return self(qstate)


@dataclass
class Hadamard(CliffordGate):
    '''
       ___
      |   |
    --| H |--
      |___|

    H I H ->  I
    H X H ->  Z
    H Y H -> -Y
    '''
    target: int
    __slots__ = ('target',)

    def transform_generator(self, g):
        n = self.target
        src = g[n]
        # swap XZ
        g[n] = (src & QState.X and QState.Z) | (src & QState.Z and QState.X)
        return 1 if src == QState.Y else 0

    def inverse(self, qstate):
        return self(qstate)


@dataclass
class Phase(CliffordGate):
    '''
       ___
      |   |
    --| S |--
      |___|

    S {I,Z} S' ->  {I,Z}
    S   X   S' ->    Y
    S   Y   S' ->   -X
    '''
    target: int
    __slots__ = ('target',)

    def transform_generator(self, g):
        n = self.target
        src = g[n]
        # z ^= x
        g[n] ^= (src & QState.X) and QState.Z
        return 1 if src == QState.Y else 0

    def inverse(self, qstate):
        return self(
               self(
               self(qstate)))


######################################################

@dataclass
class ControlledZ(CliffordGate):
    '''
    ----O----
        |
       _|_
      |   |
    --| Z |--
      |___|

    cZ {II,ZI,IZ,ZZ} cZ ->  {II,ZI,IZ,ZZ}
    cZ    {IX,IY}    cZ ->     {ZX,ZY}
    cZ    {XI,YI}    cZ ->     {XZ,YZ}
    cZ    {XX,YX}    cZ ->    -{YY,XY}
    '''
    target: Tuple[int, int]
    __slots__ = ('target',)

    def transform_generator(self, g):
        a, b = self.target
        ga, gb = g[a], g[b]

        g[a] ^= (gb & QState.X) and QState.Z
        g[b] ^= (ga & QState.X) and QState.Z
        return 1 if ga & gb & QState.X else 0

    def inverse(self, qstate):
        return self(qstate)


@dataclass
class Swap(CliffordGate):
    '''Equivalent to CXa,b CXb,a CXa,b
    ----X----
        |
        |
    ----X----
    Swap {II,XX,YY,ZZ} Swap -> {II,XX,YY,ZZ}
    Swap   {IX,IY,IZ}  Swap ->  {XI,YI,ZI}
    Swap   {XY,XZ,YZ}  Swap ->  {YX,ZX,ZY}
    '''
    target: Tuple[int, int]
    __slots__ = ('target',)

    def transform_generator(self, g):
        a, b = self.target
        g[b], g[a] = g[a], g[b]
        return 0

    def inverse(self, qstate):
        return self(qstate)


@dataclass
class PauliZ(CliffordGate):
    '''Equivalent to 2 Phase
       ___
      |   |
    --| Z |--
      |___|

    Z {I,Z} Z ->  {I,Z}
    Z {X,Y} Z -> -{X,Y}
    '''
    target: int
    __slots__ = ('target',)

    def transform_generator(self, g):
        return 1 if g[self.target] & QState.X else 0

    def inverse(self, qstate):
        return self(qstate)


@dataclass
class PauliX(CliffordGate):
    '''Equivalent to H Z H
       ___
      |   |
    --| X |--
      |___|

    X {I,X} X ->  {I,X}
    X {Z,Y} X -> -{Z,Y}
    '''
    target: int
    __slots__ = ('target',)

    def transform_generator(self, g):
        return 1 if g[self.target] & QState.Z else 0

    def inverse(self, qstate):
        return self(qstate)


######################################################

class QCircuit(MutableSequence[Union[Measure, CliffordGate]]):
    '''For storing and instantiating quantum gates'''

    def __call__(self, qstate):
        '''Apply the gates of the circuit in-place on the given state'''
        for gate in self:
            gate(qstate)
        return qstate

    def inverse(self, qstate):
        '''Apply the inverse of the gates of the circuit
        in reverse order in-place on the given stae'''
        for gate in reversed(self):
            gate.inverse(qstate)
        return qstate

    @classmethod
    def from_str(cls, src):
        '''
        create a circuit from a string containing space separated gate specification

        Example
        --------
            >>> QCircuit.from_str("H0 CX0,1 H0")
            QCircuit([Hadamard(0), CNot((0,1)), Hadamard(0)])

        Gate Format Specification
        -------------------------
            Hn :
                Hadamard on n-th index qbit
            Sn :
                Phase on n-th index qbit
            CXa,b :
                Controlled X on qbit b with qbit a as control
            Mn :
                Measure the n-th index bit in the computational basis
            Xn :
                Pauli-X on n-th index qbit
            Zn :
                Pauli-Z on n-th index qbit
            CZa,b :
                Controlled-Z on qbit b with qbit a as control
            SWAPa,b :
                Swap bit a and b
        '''
        translate = { 
            'H': Hadamard, 'S': Phase, 'CX': ControlledNot,
            'M': Measure,  'Z': PauliZ, 'X': PauliX,
            'CZ': ControlledZ, 'SWAP': Swap
        }
        args = re.compile(r'([\d,]+)')
        gates = []
        for spec in src.upper().split():
            gate_spec, target, _ = args.split(spec)
            target = int(target) if ',' not in target else tuple(map(int, target.split(',')))
            gates.append( translate[gate_spec](target) )
        
        return cls(gates)

    def __init__(self, gates):
        self.gates = list(gates)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.gates})'
   
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(self.gates[key])
        else:
            return self.gates[key]

    def __setitem__(self, key, item):
        self.gates[key] = item

    def __delitem__(self, key):
        del self.gates[key]

    def insert(self, key, item):
        self.gates.insert(key, item)

    def __add__(self, other):
        return self.__class__(self).extend(other)

    def __len__(self):
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)
