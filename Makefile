OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}
OUTPUT_FILE=${OUTPUT}.${FORMAT}

all: ${OUTPUT_FILE}
	@echo "=> Output document up-to-date at \n\t${OUTPUT_FILE}"

${OUTPUT_FILE}: *.tex tex/ img/
	latexmk -${FORMAT} main ${JOBOPTS}

clean:
	latexmk -C ${JOBOPTS}
