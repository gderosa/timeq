# Just a simple proxy to latexmk

INPUT=main
OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}

all:
	latexmk -${FORMAT} ${INPUT} ${JOBOPTS}

clean:
	latexmk -C ${JOBOPTS}
