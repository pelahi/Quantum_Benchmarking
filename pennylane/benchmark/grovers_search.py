from time import time
from inspect import currentframe, getframeinfo
from utils import logger as lg
from scipy.sparse import csr_array
import pennylane as qml
import pennylane.numpy as np
from toolkit import *
from utils import logger as lg
from unitaries import bin_array, diagonal_pauli_decompose, phase_shift, circulant_mixer, complete_eigenvalues
 
def main_args():

    import argparse

    parser = argparse.ArgumentParser(description = "Run a test QAOA circuit with random variational parameters.", epilog = "Simulation output: last expectation value, qubits, depth, repeats, backend, graph sparsity, simulation time")
    parser.add_argument("-m", "--max_qubits", default=4, type=int, dest='max_qubits', nargs='?', help = 'number of qubits, default = 4')
    parser.add_argument("-q", "--min_qubits", default=3, type=int, dest='min_qubits', nargs='?', help = 'number of qubits, default = 3')
    parser.add_argument("-r", "--repeats", default=1, type=int, dest='repeats', nargs='?', help = 'number of expectation value evaluations, default = 1')
    # parser.add_argument("-n", "--ngates", default=100, type=int, dest='n_gates', nargs='?', help = 'number of random single qubit gates')
    # parser.add_argument("-b", "--backend", default="lightning.qubit", type=str, dest='backend', nargs='?', help = 'simulation backend, default = lightning.qubit')
    # parser.add_argument("-o", "--options", default=[],  dest='options', nargs = '*', help = 'backend-specific keyword options for device creation')
    # parser.add_argument("-s", "--seed", default=42, dest='seed', type = int, nargs = '?', help = 'seed random number generation')
    # parser.add_argument("-p", "--plot", default=False, dest='plot', action = 'store_true', help = 'plot the cicuit, do not simulate')

    return parser.parse_args()



def _decomposition_with_one_worker(control_wires, target_wire, work_wire):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.3 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a single work wire"""
    tot_wires = len(control_wires) + 2
    partition = int(np.ceil(tot_wires / 2))

    first_part = control_wires[:partition]
    second_part = control_wires[partition:]

    qml.MultiControlledX(
        wires=first_part + work_wire,
        work_wires=second_part + target_wire,
    )
    qml.MultiControlledX(
        wires=second_part + work_wire + target_wire,
        work_wires=first_part,
    )
    qml.MultiControlledX(
        wires=first_part + work_wire,
        work_wires=second_part + target_wire,
    )
    qml.MultiControlledX(
        wires=second_part + work_wire + target_wire,
        work_wires=first_part,
    )

def grovers_search(device, depth, n_expvals, seed, *args):
    qubits = len(device.wires)
    wires = [i for i in range(qubits)]

    @qml.qnode(device)
    def circuit(marked_state, depth):
        for wire in wires:
            qml.Hadamard(wire)
        for _ in range(depth):
            qml.FlipSign(marked_state, wires)
            qml.GroverOperator(wires=wires)

        return qml.probs()

    timer = lg.Timer(lg.__getcurframe__())
    rng = np.random.default_rng(seed)
    for _ in range(n_expvals):
        marked_int = rng.integers(0, 2**qubits)
        marked_state = bin_array(marked_int, qubits)
        expval = circuit(marked_state, depth)[marked_int]
    deltat = timer.get_elapsed_time()
    specs = qml.specs(circuit)
    gate_sizes = specs(marked_state,depth)['resources'].gate_sizes
    circuit_depth = specs(marked_state,depth)['resources'].depth
    return float(expval), deltat, circuit_depth, gate_sizes[1], gate_sizes[2]

def grovers_search_decomposed(device, depth, n_expvals, seed, *args):
    # TODO: why have work_wires?
    qubits = len(device.wires) - 1
    wires = [i for i in range(qubits)]
    work_wires=[qubits]

    @qml.qnode(device)
    def circuit(H, depth):
        for wire in wires:
            qml.Hadamard(wire)
        for _ in range(depth):
            phase_shift(np.pi, wires, H)
            for wire in wires[:-1]:
                qml.Hadamard(wire)
                qml.PauliX(wire)
            qml.PauliZ(wires[-1])
            _decomposition_with_one_worker(wires[:-1], [wires[-1]], work_wires)
            qml.PauliZ(wires[-1])
            for wire in wires[:-1]:
                qml.PauliX(wire)
                qml.Hadamard(wire)
        return qml.probs(wires=wires)


    rng = np.random.default_rng(seed)
    
    # grab the nano second timer 
    timer = lg.Timer(lg.__getcurframe__())
    marked_int = rng.integers(0, 2**qubits)
    H = diagonal_pauli_decompose(csr_array(([1], ([marked_int], [0])), shape = (2**qubits, 1)))
    for _ in range(n_expvals):
        expval = circuit(H, depth)[marked_int]
    deltat = timer.get_elapsed_time()
    specs = qml.specs(circuit)
    gate_sizes = specs(H,depth)['resources'].gate_sizes
    circuit_depth = specs(H,depth)['resources'].depth
    return float(expval), deltat, circuit_depth, gate_sizes[1], gate_sizes[2]

def grovers_search_qft(device, depth, n_expvals, seed, *args):

    qubits = len(device.wires)

    wires = range(qubits)

    eigen_decomp = diagonal_pauli_decompose(complete_eigenvalues(2**qubits))

    @qml.qnode(device)
    @qml.compile()
    def circuit(H, gammas_ts):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for gamma, t in zip(*qml.numpy.split(gammas_ts, 2)):
            phase_shift(gamma, wires, H)
            circulant_mixer(t, wires, eigen_decomp)
        return qml.expval(H)

    timer = lg.Timer(lg.__getcurframe__())
    rng = qml.numpy.random.default_rng(seed)
    gammas_ts = qml.numpy.array([qml.numpy.pi for i in range(depth)] + [qml.numpy.pi/2**qubits for _ in range(depth)])
    marked_int = rng.integers(0, 2**qubits)
    H = diagonal_pauli_decompose(csr_array(([1], ([marked_int], [0])), shape = (2**qubits, 1)))
    for _ in range(n_expvals):
        expval = circuit(H, gammas_ts)
    deltat = timer.get_elapsed_time()
    specs = qml.specs(circuit)
    gate_sizes = specs(H, gammas_ts)['resources'].gate_sizes
    circuit_depth = specs(H, gammas_ts)['resources'].depth
    # if logging 
    if "verbose" in args:
        report : str = "Time Taken (s) " + str(deltat)
        report += ": Circuit " + str(circuit_depth) + "," + str(gate_sizes[1]) + ' ' + str(gate_sizes[2])
        report += ": Results " + str(float(expval))
        lg.LogMemory(lg.__getcurframe__())
        lg.Log(report, lg.__getcurframe__())
    
    return float(expval), deltat, circuit_depth, gate_sizes[1], gate_sizes[2]


if __name__ == "__main__":

    # lets log the memory at the start 
    args = main_args()
    if args.min_qubits >= args.max_qubits:
        args.max_qubits = args.min_qubits + 1
    
    lg.LogMemory(lg.__getcurframe__())
    for qubits in range(args.min_qubits, args.max_qubits):
        device = qml.device("lightning.qubit", wires=qubits + 1)
        depth = int(np.sqrt(2**qubits) * np.pi / 4)
        lg.Log("Running with qubits=" + str(qubits) + " depth=" + str(depth), lg.__getcurframe__())
        results = list()
        for repeat in range(args.repeats):
            results.append(grovers_search_qft(device, depth, 1, repeat, "verbose"))
