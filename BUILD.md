# Build environment

### "Development" output

If you have a full [LaTeX](https://www.latex-project.org/) installation,
and [`make`](https://www.gnu.org/software/make/), just run
```bash
make
```
in the project root directory.

This will generate `build/dev.pdf`.

### "Production" output

Another PDF will be generated, named `prod.pdf`, with TODOs removed, by issuing
```bash
make prod
```

You may also compile (and preview) both versions for quick comparison etc.:
```bash
make dev prod
```

### Continuous build

Run
```bash
make cont
```
to continuously build and monitor files for updates (development only).

### PDF previewer

In all the above, when available, a default PDF viewer is automatically opened on the generated document.

You can force a different PDF viewer by setting the `PDF_PREVIEWER` environment variable in your system.


### Cleanup
Cleanup with
```bash
make clean
```

### If "Make" is unavailable

Read the [`Makefile`](Makefile) to see the underlying [`latexmk`](https://www.ctan.org/pkg/latexmk/) commands:
you may want to use them directly.

### Optional: [SyncTeX](https://www.tug.org/TUGboat/tb29-3/tb93laurens.pdf) examples

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
