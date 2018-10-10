from stabilizer_sim import *
from unittest import TestCase, main, mock

I, X, Y, Z = GeneralPauli.I, GeneralPauli.X, GeneralPauli.Y, GeneralPauli.Z
ket = '{}|{}⟩'.format


class GateTest(TestCase):

    def assert_gate_transform(self, gate, inputs, outputs, signflip):
        for i, o, s in zip(inputs, outputs, signflip):
            pi, po = GeneralPauli.identity(), GeneralPauli.identity()
            pi.update(enumerate(i))
            po.update(enumerate(o))
            po.phase = s
            gate.transform_generator(pi)
            self.assertEqual(pi, po)


class TestQState(TestCase):
    '''
    Initial tableau should look like:
        Destabilizer:
            +XII...I
            +IXI...I
            +IIX...I
            ........
            ........
            ........
            +III...X
        Stabilizer:
            +ZII...I
            +IZI...I
            +IIZ...I
            ........
            ........
            ........
            +III...Z

        with +|00...0⟩ state
    '''
    def test_zeros(self):
        q1 = QState.zeros(1)
        self.assertEqual(q1.ket(), ket('+','0'))
        self.assertEqual(
                str(q1),
                '\n'.join([
                    'D-',
                    '+X',
                    'S-',
                    '+Z']))
        q4 = QState.zeros(4)
        self.assertEqual(
                str(q4),
                '\n'.join([
                    'D----',
                    '+XIII',
                    '+IXII',
                    '+IIXI',
                    '+IIIX',
                    'S----',
                    '+ZIII',
                    '+IZII',
                    '+IIZI',
                    '+IIIZ']))
        self.assertEqual(q4.ket(), ket('+', '0000'))


class TestPhase(GateTest):
    '''
    {I,Z} ->  {I,Z}
      X   ->    Y
      Y   ->   -X
    '''
    def test_state(self):
        s0 = Phase(0)

        inputs   = [[I], [X], [Y], [Z]]
        outputs  = [[I], [Y], [X], [Z]]
        signflip = [  0,   0,   1,   0]

        self.assert_gate_transform(s0, inputs, outputs, signflip)


class TestHadamard(GateTest):
    '''
    I ->  I
    X ->  Z
    Y -> -Y
    '''
    def test_state(self):
        h0 = Hadamard(0)

        inputs   = [[I], [X], [Y], [Z]]
        outputs  = [[I], [Z], [Y], [X]]
        signflip = [  0,   0,   1,   0]

        self.assert_gate_transform(h0, inputs, outputs, signflip)


class TestCNot(GateTest):
    '''
    {II,ZI,IX,ZX} ->  {II,ZI,IX,ZX}
       {IZ,IY}    ->     {ZZ,ZY}
       {XI,YI}    ->     {XX,YX}
         XY       ->        YZ
         XZ       ->       -YY
    '''
    def test_single(self):
        cx01 = ControlledNot(target=(0, 1))

        inputs    = [[I, I], [Z, I], [I, X], [Z, X]]
        outputs   = [[I, I], [Z, I], [I, X], [Z, X]]
        signflip  = [     0,      0,      0,      0]

        inputs   += [[I, Z], [I, Y], [Z, Z], [Z, Y]]
        outputs  += [[Z, Z], [Z, Y], [I, Z], [I, Y]]
        signflip += [     0,      0,      0,      0]

        inputs   += [[X, I], [Y, I], [X, X], [Y, X]]
        outputs  += [[X, X], [Y, X], [X, I], [Y, I]]
        signflip += [     0,      0,      0,      0]

        inputs   += [[X, Y], [Y, Z], [X, Z], [Y, Y]]
        outputs  += [[Y, Z], [X, Y], [Y, Y], [X, Z]]
        signflip += [     0,      0,      1,      1]

        self.assert_gate_transform(cx01, inputs, outputs, signflip)


class TestMeasure(TestCase):
    '''
         Z     -> 0
       - Z     -> 1
    {+,-}{X,Y} -> ?
    '''
    def test_definite(self):
        s = QState.zeros(2)
        c = QCircuit.from_str("X0 M0 M1")
        c(s)

        self.assertEqual(c[-1].outcome, 0)
        self.assertEqual(c[-2].outcome, 1)

    def test_indefinite(self):

        def completely_random(*args):
            return 0
        with mock.patch('random.randint', completely_random):
            s = QState.zeros(2)
            c = QCircuit.from_str("H0 M0 M1")
            c(s)

            self.assertEqual(c[-1].outcome, 0)
            self.assertEqual(c[-2].outcome, 0)

        def truely_random(*args):
            return 1
        with mock.patch('random.randint', truely_random):
            s = QState.zeros(2)
            c = QCircuit.from_str("H0 M0 M1")
            c(s)
            self.assertEqual(c[-1].outcome, 0)
            self.assertEqual(c[-2].outcome, 1)

    def test_entangled(self):
        s = QState.zeros(2)
        c = QCircuit.from_str("H0 CX0,1 M0 M1")
        c(s)
        self.assertEqual(c[-1].outcome, c[-1].outcome)
        

if __name__ == '__main__':
    main()
