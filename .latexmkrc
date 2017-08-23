#!perl

# Extra file extensions for lakexmk -C
$clean_ext = 'run.xml bbl fdb_latexmk';

# Do not pollute the project root, build in a subdirectory
$out_dir = "build";

# Allow using a OS environment variable change previewer
if ($ENV{'PDF_PREVIEWER'}) {
    $pdf_previewer = "$ENV{'PDF_PREVIEWER'} \%O \%S";
}
