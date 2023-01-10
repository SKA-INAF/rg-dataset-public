#!/bin/bash

INPUTFILE=$1
##MASK_DICT="{\"sidelobe\":1,\"source\":2,\"galaxy\":3}"
##MASK_DICT="{\"spurious\":1,\"compact\":2,\"extended\":3,\"extended-multicomp\":4}"
MASK_DICT="{\"spurious\":1,\"compact\":2,\"extended\":3}"

TELESCOPE="meerkat"
PIXSIZE_X="1.5"
PIXSIZE_Y="1.5"

SCRIPTDIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo "SCRIPTDIR: $SCRIPTDIR"

while IFS="," read -r FILENAME REGION remainder
do
	
	## Extract base filename from file given in list 
	FILENAME_BASE=$(basename "$FILENAME")
	FILENAME_BASE_NOEXT="${FILENAME_BASE%.*}"

	## Set output file
	OUTFILE='mask_'"$FILENAME_BASE_NOEXT"'.fits'

	echo "FILENAME=$FILENAME"
	echo "REGION=$REGION"
	echo "OUTFILE=$OUTFILE"

	python3.6 $SCRIPTDIR/make_masks_from_regions.py --img=$FILENAME --region=$REGION --outfile=$OUTFILE \
		--mvaldict="$MASK_DICT" --split_masks \
		--telescope=$TELESCOPE --pixsize_x=$PIXSIZE_X --pixsize_y=$PIXSIZE_Y \
		--save_json \
		--save_masks

done < "$INPUTFILE"
