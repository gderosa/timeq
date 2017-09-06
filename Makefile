# A proxy to latexmk

# Output directory and PDF format are set in .latexmk

# WARNING: Some auto-build plugins enabled in your editor/IDE
# WARNING: may conflict with use of `make` and `make cont`.

# This will also open the default previewer (-pv).
dev:  # "Development" version built by default (with TODOs)
	latexmk -pv dev
	@echo 'Run "make cont" if you want to continuously build.'
	@echo 'Run "make prod" to build a production version with no TODOs.'


# Continuously build and check source files for
# changes. See http://mg.readthedocs.io/latexmk.html .
cont:  # dev only
	latexmk -pvc dev

prod:
	latexmk -pv prod
	@echo 'Production pdf out with TODOs removed'

# Clean up
clean:
	latexmk -C
