#!/usr/bin/perl
use strict;

# written by Ye, NECTEC
# for MTRSS, YTU, Myanmar

my @langs;

#foreach my $trainFile ( </home/ros/experiment/my-rk/data/train.[a-z][a-z]> )
foreach my $trainFile ( </mnt/data/personal_projects/NandarLinn-SMT-Assignment/clean-data/train.[a-z][a-z]> )

{                        
        $trainFile =~ m/train.([a-z][a-z])/;
        push @langs, $1;
}

foreach my $lang (@langs)
{ 
    `./ref2sgm.pl $lang > test.$lang.ref.sgm`;
    `./src2sgm.pl $lang > test.$lang.src.sgm`;
}
