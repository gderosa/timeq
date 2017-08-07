# A proxy to latexmk

INPUT=main
OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}

# This will also open the default previewer (-pv).
#
# Replace -pv with -pvc if you wish to continuously check source files for
# changes and recompile. See http://mg.readthedocs.io/latexmk.html .
all:
	latexmk -pv -${FORMAT} ${INPUT} ${JOBOPTS}

clean:
	latexmk -C ${JOBOPTS}
