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
    input_dir       => 'DomainTest',
    clean_dir       => 'DomainTest/cleaned',
    syllable_dir    => 'DomainTest/syllable_based_200',
    word_dir        => 'DomainTest/word_based_200',
    report_dir      => 'DomainTest/reports',
    target_tokens   => 200,
    min_words       => 2,
    min_mm_chars    => 4,
    max_repeat      => 2,
    sylbreak_py     => 'tokenization_process/tools/sylbreak/python/sylbreak.py',
    oppaword_py     => 'tokenization_process/tools/oppaWord/oppa_word.py',
    oppaword_dict   => 'tokenization_process/tools/oppaWord/data/myg2p_mypos.dict',
);

GetOptions(
    'input-dir=s'     => \$opt{input_dir},
    'clean-dir=s'     => \$opt{clean_dir},
    'syllable-dir=s'  => \$opt{syllable_dir},
    'word-dir=s'      => \$opt{word_dir},
    'report-dir=s'    => \$opt{report_dir},
    'target-tokens=i' => \$opt{target_tokens},
    'min-words=i'     => \$opt{min_words},
    'min-mm-chars=i'  => \$opt{min_mm_chars},
    'max-repeat=i'    => \$opt{max_repeat},
    'sylbreak-py=s'   => \$opt{sylbreak_py},
    'oppaword-py=s'   => \$opt{oppaword_py},
    'oppaword-dict=s' => \$opt{oppaword_dict},
) or die usage();

make_path($_) for @opt{qw(clean_dir syllable_dir word_dir report_dir)};

die "sylbreak not found: $opt{sylbreak_py}\n" unless -f $opt{sylbreak_py};
die "oppaWord not found: $opt{oppaword_py}\n" unless -f $opt{oppaword_py};
die "oppaWord dictionary not found: $opt{oppaword_dict}\n" unless -f $opt{oppaword_dict};

my @files = grep { $_ !~ m{/((cleaned|syllable_based_200|word_based_200|reports|scripts)/)} }
            sort glob("$opt{input_dir}/*.csv");
die "No DomainTest CSV files found in $opt{input_dir}\n" unless @files;

open my $summary_fh, '>', "$opt{report_dir}/domain_200_summary.tsv"
    or die "Cannot write summary: $!";
print {$summary_fh} join(
    "\t",
    qw(file rows_in rows_cleaned rejected_non_myanmar rejected_short duplicates syllable_tokens word_tokens status)
), "\n";

for my $file (@files) {
    process_file($file);
}

close $summary_fh;

sub process_file {
    my ($file) = @_;
    my $base = basename($file);
    my $clean_out = "$opt{clean_dir}/$base";
    my $syl_out = "$opt{syllable_dir}/$base";
    my $word_out = "$opt{word_dir}/$base";
    my $reject_out = "$opt{report_dir}/rejected_$base.tsv";

    my ($headers, $text_cols, $rows, $stats) = read_and_clean($file, $reject_out);
    write_csv($clean_out, $headers, $rows);

    my @texts = map { $_->[0]->[$_->[1]] // '' } map {
        my $row = $_;
        map { [$row, $_] } @$text_cols;
    } @$rows;

    my @syl_texts = tokenize_syllable(@texts);
    my @word_texts = tokenize_word(@texts);

    my ($syl_rows, $syl_count) = limited_rows($rows, $text_cols, \@syl_texts, $opt{target_tokens});
    my ($word_rows, $word_count) = limited_rows($rows, $text_cols, \@word_texts, $opt{target_tokens});

    write_csv($syl_out, $headers, $syl_rows);
    write_csv($word_out, $headers, $word_rows);

    my $status = ($syl_count == $opt{target_tokens} && $word_count == $opt{target_tokens})
        ? 'ok'
        : 'not_enough_tokens';

    print {$summary_fh} join(
        "\t",
        $base,
        @{$stats}{qw(rows_in rows_cleaned rejected_non_myanmar rejected_short duplicates)},
        $syl_count,
        $word_count,
        $status,
    ), "\n";
}

sub read_and_clean {
    my ($file, $reject_out) = @_;
    open my $in_fh, '<', $file or die "Cannot read $file: $!";
    open my $rej_fh, '>', $reject_out or die "Cannot write $reject_out: $!";
    print {$rej_fh} join("\t", qw(reason source_line text)), "\n";

    my $first = <$in_fh>;
    die "Empty CSV: $file\n" unless defined $first;
    $first =~ s/\r?\n\z//;

    my @first_fields = parse_csv_line($first);
    my $has_header = looks_like_header(@first_fields);
    my @headers = $has_header ? @first_fields : default_headers(scalar @first_fields);
    my @text_cols = detect_text_columns(\@headers, \@first_fields, $has_header);

    my @rows;
    my %seen;
    my %stats = (
        rows_in               => 0,
        rows_cleaned          => 0,
        rejected_non_myanmar  => 0,
        rejected_short        => 0,
        duplicates            => 0,
    );

    my $clean_row = sub {
        my ($fields, $line_no) = @_;
        $stats{rows_in}++;

        my $raw_key = join("\t", map { $fields->[$_] // '' } @text_cols);
        if (has_non_myanmar_text($raw_key)) {
            $stats{rejected_non_myanmar}++;
            print {$rej_fh} join("\t", 'non_myanmar', "$file:$line_no", safe_report_text($raw_key)), "\n";
            return;
        }

        for my $idx (@text_cols) {
            $fields->[$idx] = clean_text($fields->[$idx] // '');
        }

        my $key = join("\t", map { $fields->[$_] // '' } @text_cols);
        if (!is_good_sentence($key)) {
            $stats{rejected_short}++;
            print {$rej_fh} join("\t", 'short_or_bad', "$file:$line_no", $key), "\n";
            return;
        }

        if ($seen{$key}++) {
            $stats{duplicates}++;
            print {$rej_fh} join("\t", 'duplicate', "$file:$line_no", $key), "\n";
            return;
        }

        push @rows, [@$fields];
        $stats{rows_cleaned}++;
    };

    my $line_no = 1;
    $clean_row->(\@first_fields, $line_no) unless $has_header;
    while (my $line = <$in_fh>) {
        $line_no++;
        $line =~ s/\r?\n\z//;
        my @fields = parse_csv_line($line);
        $clean_row->(\@fields, $line_no);
    }

    close $in_fh;
    close $rej_fh;
    return (\@headers, \@text_cols, \@rows, \%stats);

}

sub limited_rows {
    my ($rows, $text_cols, $tokenized_texts, $limit) = @_;
    my @out_rows;
    my $count = 0;
    my $text_i = 0;

    ROW:
    for my $row (@$rows) {
        my @new = @$row;
        for my $idx (@$text_cols) {
            my @tokens = grep { length $_ } split /\s+/, $tokenized_texts->[$text_i++] // '';
            my $remain = $limit - $count;
            last ROW if $remain <= 0;
            if (@tokens > $remain) {
                @tokens = @tokens[0 .. $remain - 1];
            }
            $new[$idx] = join ' ', @tokens;
            $count += @tokens;
        }
        push @out_rows, \@new;
    }

    return (\@out_rows, $count);
}

sub tokenize_syllable {
    my (@texts) = @_;
    return () unless @texts;
    my $raw = run_file_stdin_stdout([$^X, $opt{sylbreak_py}], \@texts, 'sylbreak');
    my @lines = split /\R/, $raw, -1;
    pop @lines if @lines && $lines[-1] eq '';
    return map {
        s/\|/ /g;
        normalize_spaces($_);
    } @lines;
}

sub tokenize_word {
    my (@texts) = @_;
    return () unless @texts;
    my $raw = run_oppa_word(\@texts);
    my @lines = split /\R/, $raw, -1;
    pop @lines if @lines && $lines[-1] eq '';
    return map { normalize_spaces($_) } @lines;
}

sub run_file_stdin_stdout {
    my ($cmd, $texts, $name) = @_;
    my $tmp_dir = '/tmp/domain_test_200_tokens';
    make_path($tmp_dir);
    my $in = "$tmp_dir/in_$$\_$name.txt";
    my $out = "$tmp_dir/out_$$\_$name.txt";
    write_lines($in, @$texts);
    run_shell_command(quote_cmd(@$cmd) . ' < ' . shell_quote($in) . ' > ' . shell_quote($out));
    my $result = read_text($out);
    unlink $in;
    unlink $out;
    return $result;
}

sub run_oppa_word {
    my ($texts) = @_;
    my $tmp_dir = '/tmp/domain_test_200_tokens';
    make_path($tmp_dir);
    my $in = "$tmp_dir/in_$$\_oppa.txt";
    my $out = "$tmp_dir/out_$$\_oppa.txt";
    write_lines($in, @$texts);
    run_command(
        $^X,
        $opt{oppaword_py},
        '--dict', $opt{oppaword_dict},
        '--space-remove-mode', 'my_not_num',
        '--use-bimm-fallback',
        '--bimm-boost', '150',
        '--input', $in,
        '--output', $out,
    );
    my $result = read_text($out);
    unlink $in;
    unlink $out;
    return $result;
}

sub clean_text {
    my ($text) = @_;
    $text = '' unless defined $text;
    $text = NFC(NFKC($text));
    $text =~ s/[\x{200B}-\x{200F}\x{202A}-\x{202E}\x{2060}-\x{206F}\x{FEFF}]//g;
    $text =~ s/[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}\x{FE00}-\x{FE0F}]//g;
    $text =~ s{https?://\S+|www\.\S+}{}gi;
    $text =~ s/[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}//g;
    $text =~ s/(?:^|\s)[#@][^\s]+/ /g;
    $text =~ s/[\p{P}\p{S}]+/ /g;
    $text =~ s/([^\s])\1{$opt{max_repeat},}/$1 x $opt{max_repeat}/eg;
    return normalize_spaces($text);
}

sub has_non_myanmar_text {
    my ($text) = @_;
    $text = '' unless defined $text;
    $text = NFC(NFKC($text));
    $text =~ s/[\x{200B}-\x{200F}\x{202A}-\x{202E}\x{2060}-\x{206F}\x{FEFF}]//g;
    $text =~ s/[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}\x{FE00}-\x{FE0F}]//g;
    $text =~ s{https?://\S+|www\.\S+}{}gi;
    $text =~ s/[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}//g;
    $text =~ s/(?:^|\s)[#@][^\s]+/ /g;
    $text =~ s/[\p{Myanmar}\p{N}\p{P}\p{S}\p{Space}]+//g;
    return $text =~ /\S/ ? 1 : 0;
}

sub is_good_sentence {
    my ($text) = @_;
    my $mm_chars = () = $text =~ /\p{Myanmar}/g;
    my @words = grep { /\p{Myanmar}/ } split /\s+/, $text;
    return 0 if $mm_chars < $opt{min_mm_chars};
    return 0 if @words < $opt{min_words};
    return 1;
}

sub normalize_spaces {
    my ($text) = @_;
    $text =~ s/\r?\n/ /g;
    $text =~ s/\s+/ /g;
    $text =~ s/^\s+|\s+\z//g;
    return $text;
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

sub write_csv {
    my ($path, $headers, $rows) = @_;
    open my $fh, '>', $path or die "Cannot write $path: $!";
    print {$fh} csv_join(@$headers), "\n";
    print {$fh} csv_join(@$_), "\n" for @$rows;
    close $fh;
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

sub run_command {
    my (@cmd) = @_;
    system(@cmd) == 0 or die "Command failed: @cmd\n";
}

sub run_shell_command {
    my ($cmd) = @_;
    system($cmd) == 0 or die "Command failed: $cmd\n";
}

sub quote_cmd {
    return join ' ', map { shell_quote($_) } @_;
}

sub shell_quote {
    my ($s) = @_;
    $s =~ s/'/'"'"'/g;
    return "'$s'";
}

sub looks_like_header {
    my (@fields) = @_;
    return 1 if @fields > 1 && grep { defined $_ && $_ =~ /\A(?:text|Text|Text-MM|category|label|class|id)\z/i } @fields;
    return 1 if @fields == 1 && defined $fields[0] && $fields[0] =~ /\A(?:text|Text|Text-MM|sentence|question|answer)\z/i;
    return 0;
}

sub default_headers {
    my ($count) = @_;
    return $count == 1 ? ('Text') : map { "col$_" } 1 .. $count;
}

sub detect_text_columns {
    my ($headers, $first_fields, $has_header) = @_;
    my @cols;
    for my $i (0 .. $#$headers) {
        my $h = $headers->[$i] // '';
        push @cols, $i if $h =~ /\A(?:text|Text|Text-MM|sentence|question|answer|content|body)\z/i;
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
  perl DomainTest/scripts/domain_prepare_200_tokens.pl

Outputs:
  DomainTest/cleaned/*.csv
  DomainTest/syllable_based_200/*.csv
  DomainTest/word_based_200/*.csv
  DomainTest/reports/domain_200_summary.tsv
USAGE
}
