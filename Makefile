# A proxy to latexmk

INPUT=main
OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}
BUILDOPTS=-${FORMAT} ${INPUT} ${JOBOPTS}

# This will also open the default previewer (-pv).
all:
	latexmk -pv ${BUILDOPTS}
	@echo 'Run "make cont" if you want to continuously build.'

# Continuously build and check source files for
# changes. See http://mg.readthedocs.io/latexmk.html .
cont:
	latexmk -pvc ${BUILDOPTS}

# Clean up
clean:
	latexmk -C ${JOBOPTS}
