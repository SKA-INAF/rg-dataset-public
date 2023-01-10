#!/bin/bash


WORKDIR=$1
FILEEXT=$2
DELKEYS=$3


echo "INFO: Entering directory $WORKDIR ..."


for filename in $WORKDIR/*.$FILEEXT; do
	[ -f "$filename" ] || break
  
	echo "INFO: Stripping keywords $DELKEYS from file $filename ..."

	python strip_fits_keywords.py --img=$filename --outfile=$filename --delkeys=$DELKEYS

done
