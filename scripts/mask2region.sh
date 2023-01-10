#!/bin/bash

DATADIR=$1
CLASS_REMAP="{\"sidelobe\":\"spurious\",\"source\":\"compact\",\"galaxy\":\"extended\",\"extended-multicomp\":\"extended\", \"spurious\": \"spurious\", \"compact\":\"compact\", \"extended\":\"extended\"}"
STRIP_PATTERN="mask_"

SCRIPTDIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo "SCRIPTDIR: $SCRIPTDIR"

python3.6 $SCRIPTDIR/mask2region.py --datadir=$DATADIR --class_remap="$CLASS_REMAP" --strip_pattern="$STRIP_PATTERN"
