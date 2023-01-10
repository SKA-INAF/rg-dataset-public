#!/bin/bash

WORKDIR=$1
FILE_EXT=$2
OUTFILE_PREFIX=$3

counter=1

for filename in $WORKDIR/*.$FILE_EXT; do

  digits="000"
  if [ $counter -ge 10 ] && [ $counter -lt 100 ]
  then
    digits="00"
  elif [ $counter -ge 100 ] && [ $counter -lt 1000 ]
  then
    digits="0"
  elif [ $counter -ge 1000 ] && [ $counter -lt 10000 ]
  then
    digits=""
  fi

  filename_new="$OUTFILE_PREFIX$digits$counter"'.'"$FILE_EXT"
  echo "Modifying file $filename in $filename_new"
  mv $filename $filename_new

  counter=$((counter+1))

done
