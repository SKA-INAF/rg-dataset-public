#!/bin/bash


WORKDIR=$1
FILEEXT=$2

echo "INFO: Entering directory $WORKDIR ..."


for filename in $WORKDIR/*.$FILEEXT; do
	[ -f "$filename" ] || break
  
	echo "INFO: Stripping WCS from file $filename ..."

	delwcs -v $filename

done
