# This is just a convenience proxy to latexmk

all:
	latexmk -pdf tex/main

clean:
	latexmk -C

