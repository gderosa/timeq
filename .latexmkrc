#!perl

# Prevent some errors in Linux.
$_ = '';

# Use pdflatex
$pdf_mode = 1;

# Enable synctex
$pdflatex = 'pdflatex -file-line-error --interaction=nonstopmode -synctex=1 %O %S';

# Extra file extensions for lakexmk -C
$clean_ext = 'run.xml bbl fdb_latexmk synctex.gz';

# Do not pollute the project root, build in a subdirectory
$out_dir = 'build';

if ($ENV{'PDF_PREVIEWER'}) {
    # Allow using a OS environment variable to change previewer
    $pdf_previewer = "$ENV{'PDF_PREVIEWER'} \%O \%S";
} elsif ($^O =~ /mswin/i) {
    # On Windows, override the latexmk defaults of "acroread",
    # use OS defaults instead.
    $pdf_previewer = '%S';
    # PLEASE NOTE the above doesn't work with GitHub Shell for Windows
    # (possibly with MSys in general) as "build/*.pdf" is run as a shell script :-o
    #
    # For such reason, the above regex does not capture msys. See below
    # for a possible solution.
} elsif ($^O =~ /msys/i) {
    # In MSys, sh is available...
    $pdf_previewer = q[sh -c 'start %S'];
} elsif ($^O eq 'linux') {
    $pdf_previewer = q[xdg-open %O %S];
}
# Keep defaults for MacOS
