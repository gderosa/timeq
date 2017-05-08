OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}
OUTPUT_FILE=${OUTPUT}.${FORMAT}

all: ${FORMAT}
	@echo
	@echo "=> Output document at ${OUTPUT_FILE}."
	@echo

${FORMAT}:
	latexmk -${FORMAT} main ${JOBOPTS}

clean:
	latexmk -C ${JOBOPTS}
