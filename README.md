Stabilizer-Sim
==============

A polynomial complexity simulation of stabilizer quantum circuit
based on [arxiv:quant-ph/0406196](//arxiv.org/abs/quant-ph/0406196.pdf)

Sample Usage
------------

A system of n-qbits can be initialised to the |0⟩<sup>⊗ n</sup>

```
>>> state = QState.zeros(n)
```

The generators of the stabilizers and destabilizers of the state can be shown by printing it

```
>>> print(state)
S----
+XIII
+IXII
+IIXI
+IIIX
D----
+ZIII
+IZII
+IIZI
+IIIZ
```

The state can also be shown in the computational basis
```
>>> print(state.ket())
+|0000⟩
```

A circuit can be constructed from a string, and then applied to the state
```
>>> circuit = QCircuit.from_str("H0 CX0,1 CX0,2 CX0,3")
>>> print(circuit(state))
D----
+ZIII
+IXII
+IIXI
+IIIX
S----
+XXXX
+ZZII
+ZIZI
+ZIIZ
>>> print(state.ket())
+|0000⟩
+|1111⟩
```

The outcome of a measurement can affect the state
```
>>> measure = QCircuit.from_str("M0 M1 M2 M3")
>>> measure(s)
...
>>> print(measure)
QCircuit([Measure(target=0, outcome=0), Measure(target=1, outcome=0), Measure(target=2, outcome=0)])
```

For more information, see this [demo](/Demo.ipynb) or use the built-in `help` function

Tests
-----

The tests can be run by running on the base directory:

```
python -m unittest discover
```
