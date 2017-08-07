# timeqm
Time in QM?

## Build
If you have `make` and a full LaTeX installation, just run
```bash
make
```
in the project root directory.
A PDF output document will be generated in the `build/` directory
and,
where available,
a viewer program will be launched.

### Cleanup
Cleanup with
```bash
make clean
```

### If you don't have `make`
GNU `make` is easily available in most Unix-like platforms.

Windows users can install it, for example, from
[here](http://gnuwin32.sourceforge.net/packages/make.htm).

Alternatively, you can look into [`Makefile`](Makefile)
to see the underlying command lines you can run manually
(currently based on `latexmk`).

### LaTeX
A full LaTeX installation is assumed in this document.
