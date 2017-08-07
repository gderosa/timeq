# A proxy to latexmk

INPUT=main
OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}

all:
	latexmk -${FORMAT} ${INPUT} ${JOBOPTS}

clean:
	latexmk -C ${JOBOPTS}

# Likely to only work on MacOS. TODO: find a way to degrade gracefully
# if unavailable on other platforms, allowing ths to be part of default
# 'make'.
open:
	open build/*.pdf