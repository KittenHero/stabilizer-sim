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
    'PauliX', 'PauliZ', 'ControlledZ', 'Swap',
    'GeneralPauli'
]

from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass, InitVar
from typing import Union, Tuple, List, MutableSequence, Mapping, ClassVar
from collections import OrderedDict
from heapq import merge
from itertools import chain
import re
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log = logging.FileHandler('debug.log', mode='w')
log.setLevel(logging.DEBUG)
logger.addHandler(log)

@dataclass
class GeneralPauli(OrderedDict):
    '''For representing generalised Pauli or a stabiliser'''
    components: InitVar[List[int]]
    targets: InitVar[List[int]]
    phase: int = 0

    I: ClassVar[int] = 0b00
    Z: ClassVar[int] = 0b01
    X: ClassVar[int] = 0b10
    Y: ClassVar[int] = 0b11
    pauli_str: ClassVar[Mapping[int, str]] = { I: 'I', X: 'X', Y: 'Y', Z:'Z' }

    def __post_init__(self, components, targets):
        # allows initializing components from string
        if components == '' or any(s in components for s in 'IZXY'):
            components = [getattr(self, symbol) for symbol in components]
        self.update(zip(targets, components))

    def __imul__(self, g):
        phase = 0
        for k, (sk, gk) in self.joint_items(self, g):
            prod = sk ^ gk
            self[k] = prod
            phase = (phase + self.levi_civita_mod4(gk, sk, prod)) & 0b11
        self.phase ^= (phase >> 1) ^ g.phase
        return self

    @classmethod
    def identity(cls):
        return cls('', [])

    @classmethod
    def levi_civita_mod4(cls, *order):
        I,Z,X,Y = cls.I, cls.Z, cls.X, cls.Y
        if I in order: return 0
        if order in ((Z,X,Y), (X,Y,Z), (Y,Z,X)): return 1
        else: return 3

    @staticmethod
    def joint_items(*paulis):
        return (
            (key, [p[key] for p in paulis])
             for key in merge_uniq(*paulis)
        )

    def commute(self, other):
        '''Returns true if the Paulis commute'''
        return sum(
            self.levi_civita_mod4(gi, hi, gi ^ hi)
            for _, (gi, hi) in self.joint_items(self, other)
        ) & 0b1 == 0

    def as_str(self, n):
        return ''.join(self.pauli_str.get(self[i]) for i in range(n))
    
    def __repr__(self):
        return f'''{self.__class__.__name__}('{
            ''.join(map(self.pauli_str.get, self.values()))}', {list(self.keys())}, phase={self.phase
        })'''

    def __getitem__(self, key):
        return self.get(key, self.I)

    def __setitem__(self, key, val):
        if val != self.I:
            super().__setitem__(key, val)
            for k in list(self):
                if k > key: self.move_to_end(k)
        else:
            self.pop(key, val)
    
    def copy(self):
        return self.__class__(list(self.values()), list(self.keys()), self.phase)

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return NotImplemented
        return self.phase == other.phase and self.items() == other.items()

def merge_uniq(*items):
    prev = None
    outputs = []
    for item in merge(*items):
        if item == prev: continue
        prev = item
        outputs.append(item)
    return outputs


@dataclass
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

    stab: List['GeneralPauli']
    destab: List['GeneralPauli']
    __slots__ = ('stab', 'destab')

    @classmethod
    def zeros(cls, size):
        '''Initialize the state n qbits set to zero'''
        stab = [ GeneralPauli('Z', [r]) for r in range(size) ]
        destab = [ GeneralPauli('X', [r]) for r in range(size) ]
        return cls(stab, destab)

    def validate(self):
        '''
        Checks wheter the stabilizers and destabilizers form a complete set
        Raises
        ------
        AssertionError
            The corresponding rows of destab and stab must anti-commute
            while commuting with the rest of the generators
            Each generator must also be independent
        '''
        # corresponding pairs don't commute O(n)
        for pi, qi in zip(self.stab, self.destab):
            assert not pi.commute(qi)
        # stabilizers self commute O(n^2)
        for i, pi in enumerate(self.stab):
            for pj in stab[i + 1:]:
                assert pi.commute(pj)
        # destabilizers self commute O(n^2)
        for i, qi in enumerate(self.destab):
            for qj in destab[i + 1:]:
                assert qi.commute(qj)
        # stabilizers commute with destabilizers O(n^2)
        for i, pi in enumerate(self.stab):
            for j, qj in enumerate(self.destab):
                if i == j: continue
                assert pi.commute(qj)
        # stabilizers and destabilizers produce the full paulis
        paulis = { GeneralPauli.X, GeneralPauli.Y, GeneralPauli.Z }
        for _, gk in GeneralPauli.joint_items(*self.stab, *self.destab):
            assert len(set(gk) & paulis) > 1

    def apply_clifford(self, qgate):
        '''Apply a clifford gate on the state'''
        for i, (s, d) in enumerate(zip(self.stab, self.destab)):
            qgate.transform_generator(s)
            qgate.transform_generator(d)
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
            for i, g in enumerate(self.stab[start:], start):
                if g[col] & target: return i
            else: return None
        def gaussian_help(target, start_row=0):
            for piv_c in range(n):
                target_r = min_r(target, piv_c, start_row)
                if target_r is None: continue

                self.swap_row(target_r, start_row)

                piv_sg, piv_dg = self.stab[start_row], self.destab[start_row]
                for i in range(start_row + 1, n):
                    if not self.stab[i][piv_c] & target: continue
                    self.stab[i] *= piv_sg
                    piv_dg *= self.destab[i]
                start_row += 1
            return start_row
        x_count = gaussian_help(GeneralPauli.X)
        gaussian_help(GeneralPauli.Z, x_count)
        return x_count

    def swap_row(self, i, j):
        self.stab[i], self.stab[j] = self.stab[j], self.stab[i]
        self.destab[i], self.destab[j] = self.destab[j], self.destab[i]

    def __len__(self):
        '''The number of qbits being simulated'''
        return len(self.stab)

    def ket(self, max_ket=10):
        '''The string representation of eigenstate in ket notation'''
        n = len(self)
        x_count = self.gaussian()
        if x_count > max_ket: raise ValueError('State is too big to be written')
        def gen_basis(gen):
            phase = (sum(p == GeneralPauli.Y for p in gen.values()) + 2*gen.phase) & 0b11
            pre = {0: '+', 1: '+i', 2: '-', 3: '-i'}[phase]
            bits = ''.join('1' if g in 'XY' else '0' for g in gen.as_str(n))
            return f'{pre}|{bits}⟩'
        # This produces the 'first' ket
        gen = GeneralPauli.identity()
        for i in reversed(range(x_count, n)):
            g = self.stab[i]
            phase = g.phase
            for j in reversed(range(n)):
                if not g[j] & GeneralPauli.Z: continue
                k = j
                if gen[j] & GeneralPauli.X: phase ^= 1
            if phase: gen[k] ^= GeneralPauli.X
        bases = [gen_basis(gen)]
        # This produces the remaining x_count^2 - 1 kets
        for i in range((1 << x_count) - 1):
            ket_bits = i ^ (i + 1)
            for j in range(x_count):
                if not ket_bits & (1 << j): continue
                gen *= self.stab[j]
            bases.append(gen_basis(gen))
        return '\n'.join(bases)


    def __str__(self):
        '''matrix representation of the destabilizer and stabilizer'''
        phase_str = { 0: '+', 1: '-' }
        n = len(self)
        d_str = []
        s_str = []
        sep = '{}' + '-' * n + '\n'
        for s_gen, d_gen in zip(self.stab, self.destab):
            d_str.append(phase_str[d_gen.phase])
            d_str.extend(d_gen.as_str(n))
            d_str.append('\n')

            s_str.append(phase_str[s_gen.phase])
            s_str.extend(s_gen.as_str(n))
            s_str.append('\n')

        return sep.format('D') + ''.join(d_str) + sep.format('S') + ''.join(s_str).strip()

    def __repr__(self):
        return f'''{self.__class__.__name__}({
            self.stab},{
            self.destab,
        })'''


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
        clifford = GeneralPauli.identity()

        if all(not g[n] & GeneralPauli.X for g in qstate.stab):
            for sg, dg in zip(qstate.stab, qstate.destab):
                if not dg[n] & GeneralPauli.X: continue
                clifford *= sg
            self.outcome = clifford.phase
        else:
            for i, g in enumerate(chain(qstate.destab, qstate.stab)):
                if not g[n] & GeneralPauli.X: continue
                clifford *= g
                k = i - len(qstate)
            qstate.destab[k] = clifford
            self.outcome = random.randint(0, 1)
            qstate.stab[k] = GeneralPauli('Z', [n], self.outcome)
        return self.outcome

    def as_pauli(self):
        return GeneralPauli('Z', [self.target])

#####################################

class CliffordGate(metaclass=ABCMeta):
    '''CliffordGate acts uniformly on the generators of QState
    A quantum gate U acting on a state |ψ⟩ stabilised by a generator g ∈ S
    produces a state U|ψ⟩ = Ug|ψ⟩ = UgU'U|ψ⟩ stabilized by UgU'
    '''
    __slots__ = ()

    def __str__(self):
        '''Representation of this gate'''
        return self.symbol + (','.join(map(str, self.target)) if not isinstance(self.target, int) else str(self.target))

    def __call__(self, qstate):
        '''Apply this gate to the given state'''
        return qstate.apply_clifford(self)

    @abstractproperty
    def symbol(self):
        '''The symbolic representation of the gate'''
        pass
    
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

    @property
    def symbol(self):
        return 'CX'

    def transform_generator(self, g):
        PHASE_CHANGE = ((GeneralPauli.X, GeneralPauli.Z), (GeneralPauli.Y, GeneralPauli.Y))
        a, b = self.target
        ga, gb = g[a], g[b]

        g[a] ^= (gb & GeneralPauli.Z)
        g[b] ^= (ga & GeneralPauli.X)

        g.phase ^= 1 if (ga, gb) in PHASE_CHANGE else 0

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

    @property
    def symbol(self):
        return 'H'

    def transform_generator(self, g):
        n = self.target
        src = g[n]
        # swap XZ
        g[n] = (src & GeneralPauli.X and GeneralPauli.Z) | (src & GeneralPauli.Z and GeneralPauli.X)
        g.phase ^= 1 if src == GeneralPauli.Y else 0

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

    @property
    def symbol(self):
        return 'S'

    def transform_generator(self, g):
        n = self.target
        src = g[n]
        # z ^= x
        g[n] ^= (src & GeneralPauli.X) and GeneralPauli.Z
        g.phase ^= 1 if src == GeneralPauli.Y else 0

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
    
    @property
    def symbol(self):
        return 'CZ'

    def transform_generator(self, g):
        a, b = self.target
        ga, gb = g[a], g[b]

        g[a] ^= (gb & GeneralPauli.X) and GeneralPauli.Z
        g[b] ^= (ga & GeneralPauli.X) and GeneralPauli.Z
        g.phase ^= 1 if ga & gb & GeneralPauli.X else 0

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
    
    @property
    def symbol(self):
        return 'SWAP'

    def transform_generator(self, g):
        a, b = self.target
        g[b], g[a] = g[a], g[b]

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
    
    @property
    def symbol(self):
        return 'Z'

    def transform_generator(self, g):
        g.phase ^= 1 if g[self.target] & GeneralPauli.X else 0

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
    
    @property
    def symbol(self):
        return 'X'

    def transform_generator(self, g):
        g.phase ^= 1 if g[self.target] & GeneralPauli.Z else 0

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
    
    def __str__(self):
        return  '\n'.join(map(str, self.gates))
   
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

    def to_pbc(self, nbits):
        logger.debug(f'Transforming {self!r} to pbc')
        logger.info('Replacing basis measurements by stabilizers')
        measurements = [(i, m.as_pauli()) for i, m in enumerate(self) if isinstance(m, Measure)]

        logger.info('Commute out cliffords to the right')
        cliffords = [(i, gate) for i, gate in enumerate(self) if isinstance(gate, CliffordGate)]

        logger.warning('Conditional gates unimplemented')
        for i, gate in reversed(cliffords):
            for j, pauli in reversed(measurements):
                if j < i: break
                logger.debug(f'commuting {i}: {gate} through {j}: {pauli}')
                gate.transform_generator(pauli)
                logger.debug(f'result: {pauli}')

        logger.info('Prepending Z measurements')
        prepended = [Measure(i).as_pauli() for i in range(nbits)]

        logger.info('Commuting out anti-commuting gates')
        for i, (ii, pauli) in enumerate(measurements):
            for pz in prepended:
                if pauli.commute(pz): continue
                outcome = random.randint(0, 1)
                pauli.phase ^= outcome
                measurements[i] = (ii, outcome)
                logger.debug(f'{pauli} anti-commutes, choosen random outcome: {outcome}')
                for _, other in measurements[i + 1:]:
                    pn, pm = other.commute(pz), other.commute(pauli)
                    logger.debug(f'commuting through {other}')
                    if pn and pm: continue
                    elif not pn and not pm:
                        other.phase ^= 1
                    else:
                        other *= pz
                        other *= pauli
                        if pm: other.phase ^= 1
                    logger.debug(f'resulting {other}')
                break

        logger.info('build table for computing dependent measurements')
        logger.warning('Dependent pauli unimplemented')
        for i, (ii, m) in enumerate(measurements):
            if isinstance(m, int): continue
            copy = GeneralPauli(m.values(), m.keys(), m.phase)
            # outcome for first n bits is always 0
            for j, p in copy.items():
                if j >= nbits: break
                if p != GeneralPauli.Z: continue
                logger.debug(f'{ii}:{j} is {GeneralPauli.pauli_str[p]} setting to I')
                m[j] = GeneralPauli.I
            if m == GeneralPauli.identity():
                measurements[i] = (ii, 0)

        logger.debug(f'final result: {measurements}')
        return measurements