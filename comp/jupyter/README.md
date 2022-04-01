Notebooks here are saved as
[Python/percent](https://github.com/mwouts/jupytext#the-percent-format),
that's why you don't find the `.ipynb` but the `.py`.
This facilitates version control, thanks to the awesome
[Jupytext](https://github.com/mwouts/jupytext),
which you will need to install in order to open these files as Jupyter notebooks
(otherwise they are in any case valid Python scripts and in principle you can run them as plain Python).

You need to explicitly "Open With" Notebook in Jupyter Lab UI.

## Python environment (Anaconda/Miniconda)


To avail of latest Python (3.10 at time of writing) use the conda-forge channel.
```
conda create -n timeq -c conda-forge python=3.10
```
