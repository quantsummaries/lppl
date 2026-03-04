#!/usr/bin/bash

for i in `ls *.csv`
do
        echo $i;
        grep -v "Ticker\|Date" $i | sed 's/Price/Date/g' > ttt;
        mv ttt $i;
done
