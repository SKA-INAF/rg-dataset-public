#!/bin/bash

DATADIR=$1
##MASK_DICT="{\"sidelobe\":1,\"source\":2,\"galaxy\":3}"
MASK_DICT="{\"spurious\":1,\"compact\":2,\"extended\":3,\"extended-multicomp\":4}"

CLASS_REMAP="{\"sidelobe\":\"spurious\",\"source\":\"compact\",\"galaxy\":\"extended\"}"


TELESCOPE="meerkat"
PIXSIZE_X="1.5"
PIXSIZE_Y="1.5"

SCRIPTDIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo "SCRIPTDIR: $SCRIPTDIR"

python3.6 $SCRIPTDIR/correct_masks_and_json.py --datadir=$DATADIR \
	--mvaldict="$MASK_DICT" --class_remap="$CLASS_REMAP" \
	--telescope=$TELESCOPE --pixsize_x=$PIXSIZE_X --pixsize_y=$PIXSIZE_Y \
	--save_json \
	

