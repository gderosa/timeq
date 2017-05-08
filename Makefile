# Just a simple proxy to latexmk

INPUT=document
OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}

all:
	latexmk -${FORMAT} ${INPUT} ${JOBOPTS}

clean:
	latexmk -C ${JOBOPTS}
