#!/bin/bash

FILELIST=$1
CAESAR_DIR="/home/riggi/Analysis/SKAProjects/SKATools/CAESAR/install/macros"
SEED_THR=5
MERGE_THR=2.6
NPIX=5
DELTA_THR=0.5



while read filename 
do
	MACRO_ARGS='"'$filename'",'
	MACRO_ARGS="$MACRO_ARGS$SEED_THR"','
	MACRO_ARGS="$MACRO_ARGS$MERGE_THR"','
	MACRO_ARGS="$MACRO_ARGS$NPIX"','
	MACRO_ARGS="$MACRO_ARGS$DELTA_THR"
		

	EXE="root -l -b -q $CAESAR_DIR/MakeMask.C'(""$MACRO_ARGS"")'"

	echo "CMD: $EXE"

	eval $EXE
	
	
done < "$FILELIST"
