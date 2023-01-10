from __future__ import print_function

## STANDARD MODULES
import sys
import numpy as np
import os
import re
import json
from collections import defaultdict
import operator as op
import math

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## ASTROPY MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import regions
#from regions import read_ds9

## IMAGE PROCESSING MODULES
import cv2 as cv
import imutils
from skimage import measure
from skimage.measure import regionprops 

## GRAPHICS MODULES
#import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D


## LOGGER
import logging
import logging.config
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s - %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

regcolmap= {
	"spurious": "red",
	"extended": "yellow",
	"compact": "blue",
	"sidelobe": "red",
	"galaxy": "yellow",
	"source": "blue"
}



#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	parser.add_argument('-datadir','--datadir', dest='datadir', required=True, type=str, help='Directory containing masks & json') 
	parser.add_argument('-class_remap','--class_remap', dest='class_remap', required=False, type=str, default='',help='Class remap dictionary') 
	parser.add_argument('-strip_pattern','--strip_pattern', dest='strip_pattern', required=False, type=str, default='',help='String pattern to strip from filename') 
	
	args = parser.parse_args()	

	return args

##############
##   MAIN   ##
##############
def main():
	"""Main function"""
	
	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	datadir= args.datadir
	strip_pattern= args.strip_pattern
	class_remap_str= args.class_remap
	class_remap= {}
	if class_remap_str:
		class_remap= json.loads(class_remap_str)

	print("== class_remap ==")
	print(class_remap)

	logger.info("strip_pattern: %s" % (strip_pattern))

	#===========================
	#==   PROCESS DATA
	#===========================
	inputfiles= [os.path.join(datadir, f) for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]

	for inputfile in inputfiles:
		if not inputfile.endswith(".json"):
			continue

		inputfile_base= os.path.basename(inputfile)
		inputfile_base_noext= os.path.splitext(inputfile_base)[0]

		#regionfile= inputfile_base_noext.strip(strip_pattern) + '.reg'
		regionfile= inputfile_base_noext.replace(strip_pattern, '') + '.reg'
		logger.info("inputfile_base_noext: %s, regionfile: %s" % (inputfile_base_noext, regionfile))

		#===========================
		#==   PARSE JSON
		#===========================
		logger.info("Parsing file %s ..." % (inputfile))
		with open(inputfile, 'r') as fp:
			datadict= json.load(fp)

		#===========================
		#==   PROCESS OBJECTS
		#===========================
		objs= datadict['objs']
		regs= []

		for obj in objs:
			sname= obj['name']
			class_name_old= obj['class']
			class_name= class_remap[class_name_old]
			filename_mask= obj['mask']

			#===========================
			#==   READ MASK
			#===========================
			logger.info("Read mask image %s ..." % filename_mask)
			mask= fits.open(filename_mask)[0].data
			bmap= np.copy(mask)
			bmap[bmap>0]= 1
			bmap= bmap.astype(np.uint8)
	
			#===========================
			#==   EXTRACT CONTOURS
			#===========================
			logger.info("Find obj %s contours ..." % (sname))
			contours= cv.findContours(bmap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			contours= imutils.grab_contours(contours)
			logger.info("#%d contours found ..." % (len(contours)))

			if len(contours)<=0:
				logger.error("No contours found for mask %s (obj=%s), skip it!" % (filename_mask, sname))
				continue
	
			#===========================
			#==   CREATE REGION
			#===========================
			logger.info("Creating regions ...")
			
			meta= regions.RegionMeta({"text": sname, "tag": [class_name]})
			regcol= regcolmap[class_name]
			vis= regions.RegionVisual({"color": regcol})		

			for contour in contours:
				cshape= contour.shape
				contour= contour.reshape(cshape[0],cshape[2])
			
				x, y= np.split(contour, 2, axis=1)
				x= list(x.flatten())
				y= list(y.flatten())
			
				vertices = regions.PixCoord(x=x, y=y)
				reg = regions.PolygonPixelRegion(vertices=vertices, meta=meta, visual=vis)
				regs.append(reg)

		#===========================
		#==   SAVE REGION
		#===========================
		logger.info("Saving region to file %s ..." % (regionfile))
		regions.write_ds9(regs, regionfile, coordsys="image")
	
	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

