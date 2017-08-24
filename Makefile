# A proxy to latexmk

# $out_dir is set in .latexmk, seems more effective than -outdir

# WARNING: Some auto-build plugins enabled in your editor/IDE
# WARNING: may conflict with use of `make` and `make cont`.

BUILDOPTS=-pdf

# This will also open the default previewer (-pv).
all:
	latexmk ${BUILDOPTS} -pv
	@echo 'Run "make cont" if you want to continuously build.'

# Continuously build and check source files for
# changes. See http://mg.readthedocs.io/latexmk.html .
cont:
	latexmk ${BUILDOPTS} -pvc

# Clean up
clean:
	latexmk -C
