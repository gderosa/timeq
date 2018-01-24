## Requirements setup instruction for Linux/Ubuntu (adjust for your OS)

[Qubiter](https://github.com/artiste-qb-net/qubiter)
(or at least most of its [dependencies'] documentation)
relies on Anacon to manage Python versions and packages.
I would suggest installing
[Miniconda](https://conda.io/miniconda.html)
and accept PATH variable mangling when running the install script.

Then you need to install *numpy* and *pandas*:
```bash
conda install numpy pandas
```
and a Fortran library e.g.
```bash
sudo apt install libgfortran3
```

Then you need to install
[Python-CS-Decomposition](https://github.com/artiste-qb-net/Python-CS-Decomposition#installation):
```bash
git clone https://github.com/artiste-qb-net/Python-CS-Decomposition.git

cd Python-CS-Decomposition/DIST/Linux

./install_py_module.sh
```

Then from another directory you can install Qubiter itself:
```bash
git clone https://github.com/artiste-qb-net/qubiter.git

cd qubiter
```

From there, you should be able to run the unitary decomposition
Python scripts in this directory.

