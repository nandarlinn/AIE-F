#!/usr/bin/env perl
use strict;
use warnings;
use utf8;
use open qw(:std :encoding(UTF-8));

use File::Basename qw(basename);
use File::Path qw(make_path);
use Getopt::Long qw(GetOptions);
use Unicode::Normalize qw(NFC NFKC);

my %opt = (
    input_dir          => 'Dataset',
    output_dir         => 'data_cleaning_process/cleaned',
    report_dir         => 'data_cleaning_process/reports',
    min_myanmar_chars  => 4,
    min_words          => 2,
    max_repeat         => 2,
    keep_ascii_digits  => 1,
    reject_non_myanmar => 1,
    remove_duplicates  => 1,
);

GetOptions(
    'input-dir=s'         => \$opt{input_dir},
    'output-dir=s'        => \$opt{output_dir},
    'report-dir=s'        => \$opt{report_dir},
    'min-myanmar-chars=i' => \$opt{min_myanmar_chars},
    'min-words=i'         => \$opt{min_words},
    'max-repeat=i'        => \$opt{max_repeat},
    'keep-ascii-digits!'  => \$opt{keep_ascii_digits},
    'reject-non-myanmar!' => \$opt{reject_non_myanmar},
    'remove-duplicates!'  => \$opt{remove_duplicates},
) or die usage();

make_path($opt{output_dir});
make_path($opt{report_dir});

my @files = sort glob("$opt{input_dir}/*.csv");
die "No CSV files found in $opt{input_dir}\n" unless @files;

open my $summary_fh, '>', "$opt{report_dir}/cleaning_summary.tsv"
    or die "Cannot write summary: $!";
print {$summary_fh} join("\t", qw(file rows_in rows_kept non_myanmar short_or_bad duplicates empty_after_clean text_columns)), "\n";

for my $file (@files) {
    process_file($file);
}

close $summary_fh;

sub process_file {
    my ($file) = @_;
    my $base = basename($file);
    my $out  = "$opt{output_dir}/$base";
    my $rej  = "$opt{report_dir}/rejected_$base.tsv";

    open my $in_fh, '<', $file or die "Cannot read $file: $!";
    open my $out_fh, '>', $out or die "Cannot write $out: $!";
    open my $rej_fh, '>', $rej or die "Cannot write $rej: $!";

    my $first = <$in_fh>;
    return unless defined $first;
    $first =~ s/\r?\n\z//;

    my @first_fields = parse_csv_line($first);
    my $has_header = looks_like_header(@first_fields);
    my @headers = $has_header ? @first_fields : default_headers(scalar @first_fields);
    my @text_cols = detect_text_columns(\@headers, \@first_fields, $has_header);

    print {$out_fh} csv_join(@headers), "\n" if $has_header;
    print {$rej_fh} join("\t", qw(reason source_line cleaned_text)), "\n";

    my %seen;
    my %stats = (
        rows_in           => 0,
        rows_kept         => 0,
        non_myanmar       => 0,
        short_or_bad      => 0,
        duplicates        => 0,
        empty_after_clean => 0,
    );

    my $handle_row = sub {
        my ($fields, $line_no) = @_;
        $stats{rows_in}++;

        my $raw_key = join("\t", map { $fields->[$_] // '' } @text_cols);
        if ($opt{reject_non_myanmar} && has_non_myanmar_text($raw_key)) {
            $stats{non_myanmar}++;
            print {$rej_fh} join("\t", 'non_myanmar', "$base:$line_no", safe_report_text($raw_key)), "\n";
            return;
        }

        for my $idx (@text_cols) {
            $fields->[$idx] = clean_text($fields->[$idx] // '');
        }

        my $key = join("\t", map { $fields->[$_] // '' } @text_cols);
        if ($key =~ /\A\s*\z/) {
            $stats{empty_after_clean}++;
            print {$rej_fh} join("\t", 'empty_after_clean', "$base:$line_no", ''), "\n";
            return;
        }

        if (!is_good_sentence($key)) {
            $stats{short_or_bad}++;
            print {$rej_fh} join("\t", 'short_or_bad', "$base:$line_no", $key), "\n";
            return;
        }

        if ($opt{remove_duplicates} && $seen{$key}++) {
            $stats{duplicates}++;
            print {$rej_fh} join("\t", 'duplicate', "$base:$line_no", $key), "\n";
            return;
        }

        print {$out_fh} csv_join(@$fields), "\n";
        $stats{rows_kept}++;
    };

    if (!$has_header) {
        $handle_row->(\@first_fields, 1);
    }

    my $line_no = 1;
    while (my $line = <$in_fh>) {
        $line_no++;
        $line =~ s/\r?\n\z//;
        my @fields = parse_csv_line($line);
        $handle_row->(\@fields, $line_no);
    }

    print {$summary_fh} join(
        "\t",
        $base,
        @stats{qw(rows_in rows_kept non_myanmar short_or_bad duplicates empty_after_clean)},
        join(',', map { $headers[$_] // "col$_" } @text_cols),
    ), "\n";

    close $in_fh;
    close $out_fh;
    close $rej_fh;

}

sub clean_text {
    my ($text) = @_;
    $text = '' unless defined $text;

    $text = NFKC($text);
    $text = NFC($text);

    # Remove common invisible marks and zero-width noise, except ZWNJ/ZWJ are not
    # useful for this corpus after Unicode normalization.
    $text =~ s/[\x{200B}-\x{200F}\x{202A}-\x{202E}\x{2060}-\x{206F}\x{FEFF}]//g;

    # Remove emoji, pictographs, dingbats, variation selectors, and regional flags.
    $text =~ s/[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}\x{FE00}-\x{FE0F}]//g;

    # Remove URLs, emails, and social-style noise before punctuation stripping.
    $text =~ s{https?://\S+|www\.\S+}{}gi;
    $text =~ s/[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}//g;
    $text =~ s/(?:^|\s)[#@][^\s]+/ /g;

    # Normalize punctuation to spaces. This includes Myanmar punctuation, ASCII
    # punctuation, and paired quote/bracket characters.
    $text =~ s/[\p{P}\p{S}]+/ /g;

    $text =~ s/([^\s])\1{$opt{max_repeat},}/$1 x $opt{max_repeat}/eg;
    $text =~ s/\s+/ /g;
    $text =~ s/^\s+|\s+\z//g;

    return $text;
}

sub has_non_myanmar_text {
    my ($text) = @_;
    $text = '' unless defined $text;

    $text = NFKC($text);
    $text = NFC($text);
    $text =~ s/[\x{200B}-\x{200F}\x{202A}-\x{202E}\x{2060}-\x{206F}\x{FEFF}]//g;
    $text =~ s/[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}\x{FE00}-\x{FE0F}]//g;
    $text =~ s{https?://\S+|www\.\S+}{}gi;
    $text =~ s/[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}//g;
    $text =~ s/(?:^|\s)[#@][^\s]+/ /g;
    $text =~ s/[\p{Myanmar}\p{N}\p{P}\p{S}\p{Space}]+//g;

    return $text =~ /\S/ ? 1 : 0;
}

sub safe_report_text {
    my ($text) = @_;
    $text = '' unless defined $text;
    $text =~ s/\r?\n/ /g;
    $text =~ s/\t/ /g;
    $text =~ s/\s+/ /g;
    $text =~ s/^\s+|\s+\z//g;
    return $text;
}

sub is_good_sentence {
    my ($text) = @_;
    my $mm_chars = () = $text =~ /\p{Myanmar}/g;
    my @words = grep { /\p{Myanmar}/ } split /\s+/, $text;

    return 0 if $mm_chars < $opt{min_myanmar_chars};
    return 0 if @words < $opt{min_words};
    return 1;
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

sub csv_join {
    return join ',', map {
        my $v = defined $_ ? $_ : '';
        $v =~ s/"/""/g;
        ($v =~ /[",\r\n]/) ? qq("$v") : $v;
    } @_;
}

sub usage {
    return <<"USAGE";
Usage:
  perl data_cleaning_process/scripts/clean_myanmar_csv.pl [options]

Options:
  --input-dir DIR             Default: Dataset
  --output-dir DIR            Default: data_cleaning_process/cleaned
  --report-dir DIR            Default: data_cleaning_process/reports
  --min-myanmar-chars N       Default: 4
  --min-words N               Default: 2
  --max-repeat N              Default: 2
  --[no-]keep-ascii-digits    Default: keep ASCII digits
  --[no-]reject-non-myanmar   Default: reject rows containing non-Myanmar text
  --[no-]remove-duplicates    Default: remove duplicates per file
USAGE
}
