# qiskit-qudits

`qiskit-qudits` is an open-source Qiskit extension to simulate qudits using qubits.

## Installation

Install `qiskit-qudits` via `pip`:

```bash
pip install qiskit-qudits
```

> **Note:** `qiskit-aer` is not a direct dependency, but it is required to run simulations or execute the example notebook.

## Quick Start

```python
from qiskit_qudits import QuditQuantumCircuit

qc = QuditQuantumCircuit(2, 2, dim=3)
qc.h(0)
qc.sumx(0, 1)
qc.measure([0, 1], [0, 1])
```

## Documentation

For a practical walkthrough, see the [example notebook](example.ipynb). For general Qiskit inquiries, refer to the official [IBM Quantum Documentation](https://quantum.cloud.ibm.com/docs/).

## Citation

To cite this library, please use the attached [`CITATION.bib`](CITATION.bib) file.

## License

Distributed under the [AGPL v3 License](LICENSE), except for Qiskit stubs, which retain their original [Apache 2.0 License](LICENSE-APACHE).