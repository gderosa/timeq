Notebooks here are saved as
[Python/percent](https://github.com/mwouts/jupytext#the-percent-format),
that's why you don't find the `.ipynb` but the `.py`.
This facilitates version control, thanks to the awesome
[Jupytext](https://github.com/mwouts/jupytext),

Jupytext is now installed with conda, and it is part of the `timeq` conda environment.

## Python environment (Anaconda/Miniconda)

Download and install:

https://docs.conda.io/en/latest/miniconda.html

Setup the environment
```
conda env create --name timeq --file environment.yml
```

The file `environment.yml` was originally created by stripping down information from the output of:
```
conda env export
```

Keep the environment up to date with bugfixes etc. with:
```
conda env update --name timeq --file environment.yml
```

Optionally, this may be useful in some cases:
```
conda config --set auto_activate_base false
```

After all, you just want the `timeq` environment, which you will explicitly activate anyway:
```
conda activate timeq
```
