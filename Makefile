# A proxy to latexmk

INPUT=main
OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}
BUILDOPTS=-${FORMAT} ${INPUT} ${JOBOPTS}

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
	latexmk ${JOBOPTS} -C
