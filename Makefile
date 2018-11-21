# A proxy to latexmk

# Output directory and PDF format are set in .latexmk

# WARNING: Some auto-build plugins enabled in your editor/IDE
# WARNING: may conflict with use of `make` and `make cont`.

default: dev

cont: devc

view: devv

dev:
	latexmk dev

devv:
	latexmk -pv dev

devc:
	latexmk -pvc dev

# prod = no-TODOs

prod:
	latexmk prod

prodv:
	latexmk -pv prod

prodc:
	latexmk -pvc prod

# Posters
# poster is also the name of a directory...
poster: poster-nopv
poster-nopv:
	latexmk poster

posterv:
	latexmk -pv poster

posterc:
	latexmk -pvc poster

# Clean up
clean:
	latexmk -C
