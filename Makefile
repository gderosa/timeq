# A proxy to latexmk

# $out_dir is set in .latexmk, seems more effective than -outdir

# WARNING: Don't use `make`, and particularly `make cont`
# WARNING: if auto-build plugins are enabled in your editor/IDE

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
