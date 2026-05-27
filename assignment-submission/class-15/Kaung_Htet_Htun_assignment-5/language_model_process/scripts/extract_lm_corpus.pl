#!/usr/bin/env perl
use strict;
use warnings;
use utf8;
use open qw(:std :encoding(UTF-8));

use File::Path qw(make_path);
use Getopt::Long qw(GetOptions);

my %opt = (
    input_dir  => 'tokenization_process/word_based',
    output     => 'language_model_process/corpus/word.txt',
    min_tokens => 2,
);

GetOptions(
    'input-dir=s'  => \$opt{input_dir},
    'output=s'     => \$opt{output},
    'min-tokens=i' => \$opt{min_tokens},
) or die usage();

my @files = sort glob("$opt{input_dir}/*.csv");
die "No CSV files found in $opt{input_dir}\n" unless @files;

my ($out_dir) = $opt{output} =~ m{^(.+)/[^/]+$};
make_path($out_dir) if defined $out_dir && length $out_dir;

open my $out_fh, '>', $opt{output} or die "Cannot write $opt{output}: $!";

my %seen;
my %stats = (
    files       => scalar(@files),
    rows        => 0,
    kept        => 0,
    duplicates  => 0,
    too_short   => 0,
);

for my $file (@files) {
    open my $in_fh, '<', $file or die "Cannot read $file: $!";

    my $first = <$in_fh>;
    next unless defined $first;
    $first =~ s/\r?\n\z//;

    my @first_fields = parse_csv_line($first);
    my $has_header = looks_like_header(@first_fields);
    my @headers = $has_header ? @first_fields : default_headers(scalar @first_fields);
    my @text_cols = detect_text_columns(\@headers, \@first_fields, $has_header);

    handle_row(\@first_fields, \@text_cols) unless $has_header;
    while (my $line = <$in_fh>) {
        $line =~ s/\r?\n\z//;
        my @fields = parse_csv_line($line);
        handle_row(\@fields, \@text_cols);
    }

    close $in_fh;
}

close $out_fh;

print join("\t", qw(files rows kept duplicates too_short output)), "\n";
print join("\t", @stats{qw(files rows kept duplicates too_short)}, $opt{output}), "\n";

sub handle_row {
    my ($fields, $text_cols) = @_;
    $stats{rows}++;

    for my $idx (@$text_cols) {
        my $text = normalize_line($fields->[$idx] // '');
        my @tokens = grep { length $_ } split /\s+/, $text;

        if (@tokens < $opt{min_tokens}) {
            $stats{too_short}++;
            next;
        }

        my $line = join ' ', @tokens;
        if ($seen{$line}++) {
            $stats{duplicates}++;
            next;
        }

        print {$out_fh} $line, "\n";
        $stats{kept}++;
    }
}

sub normalize_line {
    my ($text) = @_;
    $text =~ s/\r?\n/ /g;
    $text =~ s/\s+/ /g;
    $text =~ s/^\s+|\s+\z//g;
    return $text;
}

sub looks_like_header {
    my (@fields) = @_;
    return 1 if @fields > 1 && grep { defined $_ && $_ =~ /\A(?:text|Text-MM|category|label|class|id)\z/i } @fields;
    return 1 if @fields == 1 && defined $fields[0] && $fields[0] =~ /\A(?:text|Text-MM|sentence|question|answer)\z/i;
    return 0;
}

sub default_headers {
    my ($count) = @_;
    return $count == 1 ? ('text') : map { "col$_" } 1 .. $count;
}

sub detect_text_columns {
    my ($headers, $first_fields, $has_header) = @_;
    my @cols;
    for my $i (0 .. $#$headers) {
        my $h = $headers->[$i] // '';
        push @cols, $i if $h =~ /text|sentence|question|answer|content|body|Text-MM/i;
    }
    if (!@cols) {
        my @sample = $has_header ? () : @$first_fields;
        for my $i (0 .. $#sample) {
            push @cols, $i if ($sample[$i] // '') =~ /\p{Myanmar}/;
        }
    }
    @cols = (0) unless @cols;
    return @cols;
}

sub parse_csv_line {
    my ($line) = @_;
    my @fields;
    my $field = '';
    my $in_quotes = 0;
    my @chars = split //, $line;

    for (my $i = 0; $i < @chars; $i++) {
        my $ch = $chars[$i];
        if ($in_quotes) {
            if ($ch eq '\\' && defined $chars[$i + 1] && $chars[$i + 1] eq '"') {
                $field .= '"';
                $i++;
            }
            elsif ($ch eq '"') {
                if (defined $chars[$i + 1] && $chars[$i + 1] eq '"') {
                    $field .= '"';
                    $i++;
                }
                else {
                    $in_quotes = 0;
                }
            }
            else {
                $field .= $ch;
            }
        }
        else {
            if ($ch eq '"') {
                $in_quotes = 1;
            }
            elsif ($ch eq ',') {
                push @fields, $field;
                $field = '';
            }
            else {
                $field .= $ch;
            }
        }
    }
    push @fields, $field;
    return @fields;
}

sub usage {
    return <<"USAGE";
Usage:
  perl language_model_process/scripts/extract_lm_corpus.pl \\
    --input-dir tokenization_process/word_based \\
    --output language_model_process/corpus/word.txt

Options:
  --input-dir DIR     Tokenized CSV directory
  --output FILE       Output plain text corpus
  --min-tokens N      Default: 2
USAGE
}
