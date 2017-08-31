## Build
If you have a full LaTeX installation, just run
```bash
latexmk
```
in the project root directory.
A PDF output document will be generated in the `build/` directory.

### PDF previewer

To automatically open the default PDF viewer after build:
```bash
latexmk -pv
```
You can choose a different PDF viewer by setting the `PDF_PREVIEWER` environment variable in your system.

### Continuous build

Run
```
latexmk -pvc
```
to continuously build and monitor files for updates.

### Cleanup
Cleanup with
```bash
latexmk -C
```

### "Make"

If you have `make` in your system, you can use convenient shortcuts for the above operations
like `make` for build and preview, `make cont` for continuous build, and `make clean` to cleanup;
see `Makefile` for more details.

### Windows

A [Cygwin](https://www.cygwin.com/) environment,
with `make`, `git`, `perl` and maybe `vim`,
plus [MikTeX](https://miktex.org/),
seems at the moment the best combination to build this project from the command line on Windows.

### Conflicts

Don't use `latexmk` or `make` from the command line
(especially consinuous build)
if you have an auto-build plugin or functionality
enabled in your editor or IDE.

Either disable the functionality, or use it exclusively (especially if based on `latexmk`).
