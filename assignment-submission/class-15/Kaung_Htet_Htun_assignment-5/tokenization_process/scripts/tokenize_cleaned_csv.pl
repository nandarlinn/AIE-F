#!/usr/bin/env perl
use strict;
use warnings;
use utf8;
use open qw(:std :encoding(UTF-8));

use File::Basename qw(basename);
use File::Path qw(make_path);
use Getopt::Long qw(GetOptions);

my %opt = (
    input_dir       => 'data_cleaning_process/cleaned',
    output_dir      => 'tokenization_process/syllable_based',
    report_dir      => 'tokenization_process/reports',
    mode            => 'syllable',
    delimiter       => ' ',
    sylbreak_py     => 'tokenization_process/tools/sylbreak/python/sylbreak.py',
    oppaword_py     => 'tokenization_process/tools/oppaWord/oppa_word.py',
    oppaword_dict   => 'tokenization_process/tools/oppaWord/data/myg2p_mypos.dict',
    myword_py       => 'tokenization_process/tools/myWord/myword.py',
);

GetOptions(
    'input-dir=s'   => \$opt{input_dir},
    'output-dir=s'  => \$opt{output_dir},
    'report-dir=s'  => \$opt{report_dir},
    'mode=s'        => \$opt{mode},
    'delimiter=s'   => \$opt{delimiter},
    'sylbreak-py=s' => \$opt{sylbreak_py},
    'oppaword-py=s' => \$opt{oppaword_py},
    'oppaword-dict=s' => \$opt{oppaword_dict},
    'myword-py=s'   => \$opt{myword_py},
) or die usage();

die "--mode must be syllable or word\n" unless $opt{mode} =~ /\A(?:syllable|word)\z/;

make_path($opt{output_dir});
make_path($opt{report_dir});

my @files = sort glob("$opt{input_dir}/*.csv");
die "No CSV files found in $opt{input_dir}\n" unless @files;

open my $summary_fh, '>', "$opt{report_dir}/tokenization_${\($opt{mode})}_summary.tsv"
    or die "Cannot write summary: $!";
print {$summary_fh} join("\t", qw(file rows text_columns tokens)), "\n";

for my $file (@files) {
    tokenize_file($file);
}

close $summary_fh;

sub tokenize_file {
    my ($file) = @_;
    my $base = basename($file);
    my $out = "$opt{output_dir}/$base";

    open my $in_fh, '<', $file or die "Cannot read $file: $!";
    open my $out_fh, '>', $out or die "Cannot write $out: $!";

    my $first = <$in_fh>;
    return unless defined $first;
    $first =~ s/\r?\n\z//;

    my @first_fields = parse_csv_line($first);
    my $has_header = looks_like_header(@first_fields);
    my @headers = $has_header ? @first_fields : default_headers(scalar @first_fields);
    my @text_cols = detect_text_columns(\@headers, \@first_fields, $has_header);

    my @rows;
    my @text_refs;
    my @texts;

    my $collect_row = sub {
        my ($fields) = @_;
        push @rows, $fields;
        for my $idx (@text_cols) {
            push @text_refs, [$fields, $idx];
            push @texts, $fields->[$idx] // '';
        }
    };

    $collect_row->(\@first_fields) unless $has_header;
    while (my $line = <$in_fh>) {
        $line =~ s/\r?\n\z//;
        my @fields = parse_csv_line($line);
        $collect_row->(\@fields);
    }

    my @tokenized = tokenize_texts(@texts);
    die "Tokenizer returned different line count for $base\n" unless @tokenized == @texts;

    my $token_count = 0;
    for my $i (0 .. $#tokenized) {
        my ($fields, $idx) = @{$text_refs[$i]};
        $fields->[$idx] = $tokenized[$i];
        $token_count += scalar grep { length $_ } split /\Q$opt{delimiter}\E/, $tokenized[$i];
    }

    print {$out_fh} csv_join(@headers), "\n" if $has_header;
    print {$out_fh} csv_join(@$_), "\n" for @rows;

    print {$summary_fh} join(
        "\t",
        $base,
        scalar(@rows),
        join(',', map { $headers[$_] // "col$_" } @text_cols),
        $token_count,
    ), "\n";

    close $in_fh;
    close $out_fh;
}

sub tokenize_texts {
    my (@texts) = @_;
    return () unless @texts;

    if ($opt{mode} eq 'syllable') {
        die "sylbreak not found at $opt{sylbreak_py}. Run tokenization_process/scripts/setup_tokenizers.sh first.\n"
            unless -f $opt{sylbreak_py};
        my $raw = run_file_tokenizer(
            [$^X, $opt{sylbreak_py}],
            \@texts,
            'sylbreak',
        );
        my @lines = split /\R/, $raw, -1;
        pop @lines if @lines && $lines[-1] eq '';
        return map { normalize_sylbreak_output($_) } @lines;
    }

    if (-f $opt{oppaword_py}) {
        die "oppaWord dictionary not found at $opt{oppaword_dict}\n" unless -f $opt{oppaword_dict};
        my $raw = run_file_tokenizer(
            [
                $^X, $opt{oppaword_py},
                '--dict', $opt{oppaword_dict},
                '--space-remove-mode', 'my_not_num',
                '--use-bimm-fallback',
                '--bimm-boost', '150',
            ],
            \@texts,
            'oppaWord',
        );
        my @lines = split /\R/, $raw, -1;
        pop @lines if @lines && $lines[-1] eq '';
        return map { trim($_) } @lines;
    }

    if (-f $opt{myword_py}) {
        my $raw = run_file_tokenizer(
            [$^X, $opt{myword_py}, 'word', '--delimiter', $opt{delimiter}],
            \@texts,
            'myWord',
        );
        my @lines = split /\R/, $raw, -1;
        pop @lines if @lines && $lines[-1] eq '';
        return map { trim($_) } @lines;
    }

    die "oppaWord not found at $opt{oppaword_py}. If oppaWord is unavailable, install myWord at $opt{myword_py} as the documented fallback.\n";
}

sub run_file_tokenizer {
    my ($cmd, $texts, $name) = @_;
    my $tmp_dir = '/tmp/myanmar_tokenization_process';
    make_path($tmp_dir);
    my $in = "$tmp_dir/in_$$\_$name.txt";
    my $out = "$tmp_dir/out_$$\_$name.txt";

    write_lines($in, @$texts);
    if ($name eq 'sylbreak') {
        run_shell_command(quote_cmd(@$cmd) . ' < ' . shell_quote($in) . ' > ' . shell_quote($out));
    }
    elsif ($name eq 'oppaWord') {
        run_command(@$cmd, '--input', $in, '--output', $out);
    }
    else {
        run_command(@$cmd, $in, $out);
    }
    my $tokenized = read_text($out);
    unlink $in;
    unlink $out;
    return $tokenized;
}

sub run_shell_command {
    my ($cmd) = @_;
    system($cmd) == 0 or die "Tokenizer command failed: $cmd\n";
}

sub quote_cmd {
    return join ' ', map { shell_quote($_) } @_;
}

sub shell_quote {
    my ($s) = @_;
    $s =~ s/'/'"'"'/g;
    return "'$s'";
}

sub run_command {
    my (@cmd) = @_;
    system(@cmd) == 0 or die "Tokenizer command failed: @cmd\n";
}

sub write_lines {
    my ($path, @lines) = @_;
    open my $fh, '>', $path or die "Cannot write $path: $!";
    print {$fh} join("\n", @lines), "\n";
    close $fh;
}

sub read_text {
    my ($path) = @_;
    open my $fh, '<', $path or die "Cannot read $path: $!";
    local $/;
    my $text = <$fh> // '';
    close $fh;
    return $text;
}

sub normalize_sylbreak_output {
    my ($text) = @_;
    $text =~ s/\|/$opt{delimiter}/g;
    return trim($text);
}

sub trim {
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
  perl tokenization_process/scripts/tokenize_cleaned_csv.pl --mode syllable
  perl tokenization_process/scripts/tokenize_cleaned_csv.pl --mode word --output-dir tokenization_process/word_based

Options:
  --input-dir DIR      Default: data_cleaning_process/cleaned
  --output-dir DIR     Default: tokenization_process/syllable_based
  --report-dir DIR     Default: tokenization_process/reports
  --mode MODE          syllable or word
  --delimiter STR      Default: single space
  --sylbreak-py PATH   Default: tokenization_process/tools/sylbreak/python/sylbreak.py
  --oppaword-py PATH   Default: tokenization_process/tools/oppaWord/oppa_word.py
  --oppaword-dict PATH Default: tokenization_process/tools/oppaWord/data/myg2p_mypos.dict
  --myword-py PATH     Fallback: tokenization_process/tools/myWord/myword.py
USAGE
}
