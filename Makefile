# A proxy to latexmk

# Output directory and PDF format are set in .latexmk

# WARNING: Some auto-build plugins enabled in your editor/IDE
# WARNING: may conflict with use of `make` and `make cont`.

# targets are not existing files, so no need of .PHONY
# The check for file timestamps is left to latexmk
default:
	latexmk main

view:
	latexmk -pv main

cont:
	latexmk -pvc main

c: cont

v: view

# Posters
# poster is also the name of a directory...
poster: poster-nopv
poster-nopv:
	latexmk poster

posterv:
	latexmk -pv poster

posterc:
	latexmk -pvc poster

poster-detect: poster-detect-cont # continuous onloy for now...

poster-detect-cont:
	latexmk -pvc poster-detect

# Clean up
clean:
	latexmk -C
