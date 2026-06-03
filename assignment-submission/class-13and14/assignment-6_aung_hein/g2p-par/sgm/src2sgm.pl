
#!/usr/bin/perl
use strict;

# written by Ye, NECTEC
# for MTRSS, YTU, Myanmar

my $src = shift;

print "<srcset setid=\"Myanmar_G2P_v2\" srclang=\"any\">\n";
print "<doc docid=\"exp_1_normalized\" genre=\"8000\" origlang=\"$src\">\n";

#open FILE, "/home/ros/experiment/my-rk/data/myph.$src" or die;
open FILE, "../test_clean.$src" or die;

my $id=1;

while( <FILE> )
{
	chomp;
	
	print "<seg id=\"$id\">$_ </seg>\n";
	$id++;
}

print "</doc>\n</srcset>\n";
