# A proxy to latexmk

INPUT=main
OUTPUT=build/timeqm
FORMAT=pdf
JOBOPTS=-jobname=${OUTPUT}
BUILDOPTS=-${FORMAT} ${INPUT} ${JOBOPTS}
CLEANROOT=latexmk -C; rm -f *.fdb_latexmk

# This will also open the default previewer (-pv).
all:
	latexmk ${BUILDOPTS} -pv
	${CLEANROOT}
	@echo 'Run "make cont" if you want to continuously build.'

# Continuously build and check source files for
# changes. See http://mg.readthedocs.io/latexmk.html .
cont:
	latexmk ${BUILDOPTS} -pvc
	${CLEANROOT}

# Clean up
clean:
	latexmk ${JOBOPTS} -C
	${CLEANROOT}

# Extra latexmk -C above (with no additional options) are to
# remove file created in the root, even if they were expected
# to be created in build/ only, maybe a bug, bit it seems to happen
# only when VSCode is open on the dir (not sure about other editors).
