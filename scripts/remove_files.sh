#!/bin/bash

FILELIST=$1

while read filename; do

  echo "Removing file $filename ..."
  rm $filename

done < $FILELIST
