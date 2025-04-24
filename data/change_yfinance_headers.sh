#!/usr/bin/bash

for i in `ls *.csv`
do
        echo $i;
        sed -i 's/DATE/Date/g' $i;
        sed -i 's/CLOSE/Close/g' $i;
        sed -i 's/HIGH/High/g' $i;
        sed -i 's/LOW/Low/g' $i;
        sed -i 's/OPEN/Open/g' $i;
        sed -i 's/VOLUME/Volume/g' $i;
done
