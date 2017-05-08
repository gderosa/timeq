# Just a simple proxy to latexmk

OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}

all:
	latexmk -${FORMAT} main ${JOBOPTS}

clean:
	latexmk -C ${JOBOPTS}
