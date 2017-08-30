#!perl

# Use pdflatex
$pdf_mode = 1;

# Enable synctex
$pdflatex = "pdflatex -file-line-error -synctex=1 \%O \%S";
# (cmd+shift+click on Skim for Mac, and set text editor)

# Extra file extensions for lakexmk -C
$clean_ext = 'run.xml bbl fdb_latexmk';

# Do not pollute the project root, build in a subdirectory
$out_dir = "build";

if ($ENV{'PDF_PREVIEWER'}) {
    # Allow using a OS environment variable to change previewer
    $pdf_previewer = "$ENV{'PDF_PREVIEWER'} \%O \%S";
} elsif ($^O =~ /mswin/i) {
    # On Windows, override the latexmk defaults of "acroread",
    # use OS defaults instead.
    $pdf_previewer = '%S';
    # PLEASE NOTE the above doesn't work with GitHub Shell for Windows
    # (maybe with MSys in general?) as "build/*.pdf" is run as a shell script :-o
    #
    # For such reason, the above regex does not capture msys.
    #
    # Better to use a CygWin environment or even plain PowerShell.
}
# Else keep whatever latexmk defaults for Linux or MacOS
