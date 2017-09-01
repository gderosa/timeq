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

Double-click on a point in the PDF and you will get at the editor at the corresponding line in source.

#### ...and Skim on MacOS

Inverse search: in [Skim](http://skim-app.sourceforge.net):
*Skim* -> *Preferences...* -> *Sync*:

*PDF-TeX Sync support:*

* *Preset*: `custom`
* *Command:* `code` (or full path to `code.exe`)
* *Arguments:* `--goto "%file":%line`

Cmd+Shift+click on a point in the PDF and you will get at the editor at the corresponding line in source.

### Known conflicts and limitations

#### IDE/editor autobuild

Don't use `latexmk` or `make` from the command line
(especially consinuous build)
if you have an auto-build plugin or functionality
enabled in your editor or IDE.

Either disable the functionality, or use it exclusively (especially if based on `latexmk`).

#### Windows and synctex

On Windows, if you use [TeX Live](https://www.tug.org/texlive/),
you will have a `synctex` binary, for forward search
(from source to PDF), but `latexmk` won't work under a MSys shell / GitHub shell:
use Cygwin or a plain Powershell or Cmd in this case.

If you use [MikTex](https://miktex.org/about), MSys will work,
but you won't avail of a `synctex` executable (inverse search will still work though).
