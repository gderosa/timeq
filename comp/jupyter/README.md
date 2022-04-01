Notebooks here are saved as
[Python/percent](https://github.com/mwouts/jupytext#the-percent-format),
that's why you don't find the `.ipynb` but the `.py`.
This facilitates version control, thanks to the awesome
[Jupytext](https://github.com/mwouts/jupytext),

Jupytext is now installed with conda, and it is part of the `timeq` conda environment.

## Python environment (Anaconda/Miniconda)

Download and install:

https://docs.conda.io/en/latest/miniconda.html

To avail of latest Python (3.10 at time of writing) use the [conda-forge](https://conda-forge.org/) channel.
```
conda create -n timeq -c conda-forge python=3.10
```

This may be useful in some cases:
```
conda config --set auto_activate_base false
```

After all, you want the `timeq` environment, which you will explicitly activate anyway:
```
conda activate timeq
```
