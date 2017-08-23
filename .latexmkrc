#!perl

$clean_ext = 'run.xml bbl fdb_latexmk';

if ($ENV{'PDF_PREVIEWER'}) {
    $pdf_previewer = "$ENV{'PDF_PREVIEWER'} \%O \%S";
}
