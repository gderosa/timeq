# Build environment for PDF documents

A full LaTeX env with [LaTeXMk](https://www.ctan.org/pkg/latexmk) is required.


## LaTeXMk commands to build thesis and posters

PDF outputs will be created under the `build/` directory.

```
latexmk
```

or

```
latexmk poster/<name>  # Any .`tex` file name in `poster/` dir
```

### PDF previewer and continuous build

Add `-pv` option to the `latexmk` command above to automatically launch a PDF viewer on the
result.

Or `-pvc` to enable continuous build.

You can force a different PDF viewer by setting the `PDF_PREVIEWER` environment variable in your system.

### Speed-up (development, optional)

Add argument ` opt/dev.tex`. This creates large PDF output file but save time on compression.
Not suitable for sharing.

In following execution (if no bibiliography changes), add also ` -nobibtex`.

Example:
```
latexmk opt/dev.tex
latexmk -pvc -nobibtex opt/dev.tex
```

### Cleanup

`latexmk -C` or `latexmk -C <name>` (as above), or even `rm build/*`.

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
* *Command:* `code` (or full path to `code` executable)
* *Arguments:* `--goto "%file":%line`

Cmd+Shift+click on a point in the PDF and you will get at the editor at the corresponding line in source.

#### ...and Okular on Linux

Inverse search: in [Okular](https://okular.kde.org/):
*Settings* -> *Configure Okular...* -> *Editor*:

* *Editor*: `Custom Text Editor`
* *Command:* `code --goto %f:%l:%c`

Shift+click on a point in the PDF and you will get the editor at the corresponding line in source.

### Known conflicts and limitations

#### IDE/editor autobuild

Don't use `latexmk` from the command line
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

## Docker (experimental support)

You can run everything in a container so all package provisioning is automated.
```
docker build -t timeq .
```
Then each LaTeX command above can be preceded by `./latexdockercmd.sh` e.g.
```
./scripts/docker/latexdockercmd.sh latexmk main
```
See the other scripts in `scripts/docker` too
and https://github.com/blang/latex-docker for more info.
