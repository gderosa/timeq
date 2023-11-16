#!perl

# Prevent some errors in Linux.
$_ = '';

# Use pdflatex
$pdf_mode = 1;

# Enable synctex
$pdflatex = 'pdflatex -file-line-error -synctex=1 %O %S';

#
$silent = 0;

# Extra file extensions for lakexmk -C
$clean_ext = 'run.xml bbl fdb_latexmk synctex.gz';

# Do not pollute the project root, build in a subdirectory
$out_dir = 'build';

# Address this error/warning:
# Changed files, or newly in use since previous run(s):
#     'build/main.aux'
# Latexmk: Maximum runs of pdflatex reached without getting stable files
#
# https://tex.stackexchange.com/questions/294457/latexmk-ignore-embedded-file
#
# Also, this significantly speeds the build up (hopefully, no issues with bibliography).
# $hash_calc_ignore_pattern{'aux'} = '^';
# $hash_calc_ignore_pattern{'log'} = '^';
# $hash_calc_ignore_pattern{'bbl'} = '^';

# $max_repeat = 7;

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
