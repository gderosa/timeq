# timeqm
Time in QM?

## Build
If you have `make` and a full LaTeX installation, just run
```
make
```
in the project root directory.
A PDF output document will be generated in the `build/` directory
and,
where available,
a viewer program will be launched.

Run
```
make cont
```
to continuously build and monitor files for updates.

### Cleanup
Cleanup with
```bash
make clean
```

### PDF viewer

You can set/customise the PDF viewer via PDF_PREVIEWER environment variable.

For example, adapt something like this to your system:

```
make PDF_PREVIEWER='"C:\Program Files (x86)\Adobe\Reader 11.0\Reader\AcroRd32.exe"'

make PDF_PREVIEWER=texshop
```

but you can store the variable in the system or user shell environment and just run `make`,
once this works.


## If you don't have `make` (and when not to use it)
GNU `make` is easily available in most Unix-like platforms.

Windows users can install it, for example, from
[here](http://gnuwin32.sourceforge.net/packages/make.htm).

Alternatively, you can look into [`Makefile`](Makefile)
to see the underlying command lines you can run manually
(currently based on `latexmk`).

Or just rely on your IDE or editor (see below).

### Conflicts

Don't use `make` or `latexmk` from the command line
(especially consinuous build)
if you have an auto-build plugin or functionality
enabled in your editor or IDE.

Either disable the functionality, or use it exclusively (especially if based on `latexmk`).

## LaTeX
A full LaTeX installation is assumed in this document.
