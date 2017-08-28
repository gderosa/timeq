# A proxy to latexmk

# Output directoru and PDF format are set in .latexmk

# WARNING: Some auto-build plugins enabled in your editor/IDE
# WARNING: may conflict with use of `make` and `make cont`.

# This will also open the default previewer (-pv).
all:
	latexmk -pv
	@echo 'Run "make cont" if you want to continuously build.'

# Continuously build and check source files for
# changes. See http://mg.readthedocs.io/latexmk.html .
cont:
	latexmk -pvc

# Clean up
clean:
	latexmk -C
