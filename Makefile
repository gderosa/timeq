# This is just a convenience proxy to latexmk

all:
	latexmk -pdf main

clean:
	latexmk -C