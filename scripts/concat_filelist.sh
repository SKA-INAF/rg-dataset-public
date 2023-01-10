#!/bin/bash


file1=$1
file2=$2
file_out="concat.dat"

echo "INFO: Concatenating files $file1 and $file2  ..."

paste -d',' $file1 $file2 > $file_out
#paste -d"," $file1 $file2 | column -s $',' -t > $file_out



