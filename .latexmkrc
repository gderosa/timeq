#!perl

# Use pdflatex
$pdf_mode = 1;

# Extra file extensions for lakexmk -C
$clean_ext = 'run.xml bbl fdb_latexmk';

# Do not pollute the project root, build in a subdirectory
$out_dir = "build";

if ($ENV{'PDF_PREVIEWER'}) {
    # Allow using a OS environment variable to change previewer
    $pdf_previewer = "$ENV{'PDF_PREVIEWER'} \%O \%S";
} elsif (!system("which texworks")) {
    # system() is falsey on success.
    #
    # Try other common options:
    #
    $pdf_previewer = "texworks \%O \%S";
} else {
    $pdf_previewer = "open \%O \%S";
}
