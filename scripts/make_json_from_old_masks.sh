#!/bin/bash

INPUTFILE=$1
LABEL=$2

SCRIPTDIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo "SCRIPTDIR: $SCRIPTDIR"

while IFS="," read -r FILENAME MASK remainder
do
	
	## Extract base filename from file given in list 
	FILENAME_BASE=$(basename "$FILENAME")
	FILENAME_BASE_NOEXT="${FILENAME_BASE%.*}"

	## Set output file
	OUTFILE='mask_'"$FILENAME_BASE_NOEXT"'.json'

	echo "FILENAME=$FILENAME"
	echo "MASK=$MASK"
	echo "OUTFILE=$OUTFILE"

	##python $SCRIPTDIR/make_json_from_old_masks.py --img=$FILENAME --masks=$MASK --outfile=$OUTFILE --label=$LABEL
	python3.6 $SCRIPTDIR/make_json_from_old_masks.py --img=$FILENAME --masks=$MASK --outfile=$OUTFILE --label=$LABEL

done < "$INPUTFILE"
