#!/bin/bash

FILELIST=$1
REGION_DIR=$2

while read imgfile 
do
  # - Get base filename

  imgfile_base="${imgfile##*/}"
  imgfile_base_noext="${imgfile_base%.fits}"
  filepath=`dirname $imgfile`

  regionfile="$REGION_DIR/$imgfile_base_noext"'.reg'

  #echo "imgfile_base_noext: $imgfile_base_noext"
  echo "region: $regionfile"
  echo "img: $imgfile"

  ds9 $imgfile -region $regionfile -zscale -zoom to fit
  #ds9 $imgfile -region $REGION_REF -zscale -zoom to fit $imgfile -region $regionfile -zscale -zoom to fit
  ###ds9 $imgfile -region $regionfile -zscale -zoom to fit $imgfile -region $REGION_REF -zscale -zoom to fit
  ###ds9 $imgfile -region $regionfile -zscale -zoom to fit 

done < "$FILELIST"



