#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

use warnings;
use strict;

die("ERROR syntax: input-from-sgm.perl < in.sgm > in.txt")
    unless scalar @ARGV == 0;

while(my $line = <STDIN>) {
    chop($line);
    while ($line =~ /<seg[^>]+>\s*$/i) {
	my $next_line = <STDIN>;
	$line .= $next_line;
	chop($line);
    }
    while ($line =~ /<seg[^>]+>\s*(.*)\s*$/i &&
	   $line !~ /<seg[^>]+>\s*(.*)\s*<\/seg>/i) {
	my $next_line = <STDIN>;
	$line .= $next_line;
	chop($line);
    }
    if ($line =~ /<seg[^>]+>\s*(.*)\s*<\/seg>/i) {
	my $input = $1;
	$input =~ s/\s+/ /g;
	$input =~ s/^ //g;
	$input =~ s/ $//g;
	print $input."\n";
    }
}
