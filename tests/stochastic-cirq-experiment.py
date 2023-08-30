import itertools as it
import numpy as np
import time
import json
import threading
from functools import partial
import queue
import multiprocessing.pool
from hierarqcal import (
    Qcycle,
    Qmotif,
    Qinit,
    Qmask,
    Qpermute,
    Qpivot,
    plot_circuit,
    plot_motif,
    get_tensor_as_f,
    Qunitary,
)
from typing import (
    Any,
    Dict,
    Tuple,
)
from functools import reduce
from math import dist
import cirq
from cirq import protocols
from cirq.contrib.svg import SVGCircuit
import warnings
warnings.filterwarnings('ignore')


# *********************** Classical stochastic gate set ***********************
class Mixing(cirq.Gate):
    def __init__(self):
        super(Mixing, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([
            [1.0,  1.0],
            [1.0, 1.0]
        ]) / np.array([2])

    def _circuit_diagram_info_(self, args):
        return "MX"

    # TODO: Figure out how to serialize custom gate to store circuits
    def _json_dict_(self) -> Dict[str, Any]:
        return

    def _from_json_dict_(self):
        return


def MX(bits, symbols=None, circuit=None):
    mix = Mixing()
    q0 = cirq.LineQubit(bits[0])
    circuit += mix.on(q0)
    return circuit


# *********************** Quantum gate set ***********************
def CNOT(bits, symbols=None, circuit=None):
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.CNOT(q1, q0)
    return circuit


def H(bits, symbols=None, circuit=None):
    q0 = cirq.LineQubit(bits[0])
    circuit += cirq.H(q0)
    return circuit


# *********************** Helper functions ************************
def circ_dist(p, q, eps=0.001):
    """
    Computes Euclidean distance between p and q, then returns true if distance is larger than epsilon
    :param p: L1 normalised vector
    :param q: L1 normalised vector
    :param eps: scalar, threshold value
    :return: True if dist(p,q) > eps
    """
    d = abs(dist(p, q))
    if d >= eps:
        return d


def get_circs(permutation, ordering):
    """ For each ordering of bits/qubits and each possible permutation of gates, the probability vectors of each
    quantum circuit and its stochastic analogue is computed. The distance of the output vectors are computed,
    and if greater than epsilon, the respective circuits are stored.
    :param permutation: list specifying the possible gate permutations
    :param ordering: list specifying the qubit ordering
    :return: None
    """
    # Quantum circuit construction using Qmotif base objects which allows us to create all possible circuits
    # up to a given depth using graph like structures and connecting to cirq (disclaimer: maybe there is a better way)
    circuit_motif_q = [
        Qmotif(E=[permutation], mapping=q_gates[gate_ind])
        for permutation, gate_ind in zip(permutation, ordering)
    ]
    hierq_q = Qinit(qubit_indices) + reduce(lambda a, b: a + b, circuit_motif_q)
    circuit_q = hierq_q(backend="cirq")

    # Classical circuit construction
    circuit_motif_s = [
       Qmotif(E=[permutation], mapping=s_gates[gate_ind])
       for permutation, gate_ind in zip(permutation, ordering)
    ]
    hierq_s = Qinit(qubit_indices) + reduce(lambda a, b: a + b, circuit_motif_s)
    circuit_s = hierq_s(backend="cirq")

    # Obtain the probability vectors:
    # quantum
    sim = cirq.Simulator()
    results_q = sim.simulate(circuit_q)
    q = cirq.state_vector_to_probabilities(results_q.state_vector())
    # classical
    results_s = sim.simulate(circuit_s)
    s = results_s.state_vector()
    # TODO: save circuits - need to figure out how to serialize custom gate objects to json so that we can save them
    return circ_dist(q, s)


def get_results(qubit_indices, subset):
    """
    Function to compute all permutations of possible circuits
    :param qubit_indices: list index for bits/qubits
    :param subset: possible ordering of qubit indices
    :return:
    """
    orderings = set(it.permutations(subset))
    # Get all gate permutations to feed into circuit construction
    gate_permutations = dict()
    for gate in subset:
        gate_permutations[gate] = list(it.permutations(qubit_indices, 2))
    combined = dict()
    for ordering in orderings:
        combined[ordering] = list(
            it.product(*[gate_permutations[gate] for gate in ordering])
        )
    for ordering, permutations in combined.items():
        # TODO: Could also parallelize loop here
        for permutation in permutations:
            get_circs(permutation, ordering)


# *********************** Variables to test ************************
cnot = Qunitary(CNOT, n_symbols=1, arity=2)
h = Qunitary(H, n_symbols=1, arity=1)
mixing = Qunitary(MX, n_symbols=1, arity=1)
# classical gate set
s_gates = [cnot, mixing]
# quantum gate set
q_gates = [cnot, h]
# index gates
n_gates = len(q_gates)
gates_idx = {i for i in range(n_gates)}


# no of qubits
nq = 2
qubit_indices = [i for i in range(nq)]
# max circuit depth
depth = 2

# ************************ Generate all graph permutations and results************************
start = time.process_time()

# iterable to parallelize over
subsets = list(it.product(gates_idx, repeat=depth))

# Parallelize over subsets
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    res = pool.map(partial(get_results, qubit_indices), subsets)
    pool.close()

# print time taken
print('Runtime (s): ', time.process_time() - start)
