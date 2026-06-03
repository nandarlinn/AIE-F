#!/usr/bin/perl
use strict;

# written by Ye, NECTEC
# for MTRSS, YTU, Myanmar

my $src = shift;

print "<srcset setid=\"Myanmar Grapheme to Phoneme\" srclang=\"any\">\n";
print "<doc docid=\"none\" genre=\"100\" origlang=\"$src\">\n";

#open FILE, "/home/ros/experiment/my-rk/data/test.$src" or die;
open FILE, "/mnt/data/personal_projects/NandarLinn-SMT-Assignment/clean-data/test.$src" or die;

my $id=1;

while( <FILE> )
{
	chomp;
	
	print "<seg id=\"$id\">$_ </seg>\n";
	$id++;
}

print "</doc>\n</srcset>\n";
