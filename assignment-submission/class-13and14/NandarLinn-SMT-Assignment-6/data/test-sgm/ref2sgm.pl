#!/usr/bin/perl
use strict;

# written by Ye, NECTEC
# for MTRSS, YTU, Myanmar

my $trg = shift;

#print "<refset trglang=\"$trg\" setid=\"sl_data\" srclang=\"any\">\n";
print "<refset trglang=\"$trg\" setid=\"Myanmar Grapheme to Phoneme\" srclang=\"any\">\n";
print "<doc sysid=\"ref\" docid=\"none\" genre=\"100\" origlang=\"any\">\n";

#open FILE, "/home/ros/experiment/my-rk/data/test.$trg" or die;
open FILE, "/mnt/data/personal_projects/NandarLinn-SMT-Assignment/data/test.$trg" or die;
             
my $id=1;

while( <FILE> )
{
	chomp;
	
	print "<seg id=\"$id\">$_ </seg>\n";
	$id++;
}

print "</doc>\n</refset>\n";
