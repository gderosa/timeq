## Build and development environment

If you have a full [LaTeX](https://www.latex-project.org/) installation,
including [latexmk](https://www.ctan.org/pkg/latexmk/),
just run
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

If you have [`make`](https://www.gnu.org/software/make/) in your system, you can use convenient shortcuts for the above operations
like `make` for build and preview, `make cont` for continuous build, and `make clean` to cleanup;
see [`Makefile`](Makefile) for more details.

### [SyncTeX](https://www.tug.org/TUGboat/tb29-3/tb93laurens.pdf) examples

#### Visual Studio Code and SumatraPDF on Windows

Inverse search: in [SumatraPDF](https://www.sumatrapdfreader.org/free-pdf-reader.html):
*Settings* -> *Options* -> *Enter the command line to invoke when you double-click on the PDF document*:
```
cmd /c code --goto %f:%l:%c
```
(assuming VSCode is in the PATH &mdash; replace `code` with full path otherwise).

#### ...and Skim on MacOS

Inverse search: in [Skim](http://skim-app.sourceforge.net):
*Skim* -> *Preferences...* -> *Sync*:

*PDF-TeX Sync support:*

* *Preset*: `custom`
* *Command:* `code` (or full path to `code.exe`)
* *Arguments:* `--goto "%file":%line`

### Conflicts

Don't use `latexmk` or `make` from the command line
(especially consinuous build)
if you have an auto-build plugin or functionality
enabled in your editor or IDE.

Either disable the functionality, or use it exclusively (especially if based on `latexmk`).
