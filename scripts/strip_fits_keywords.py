## STANDARD MODULES
import sys
import numpy as np
import os
import re
import json
from collections import defaultdict
import operator as op
import copy

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## ASTROPY MODULES
from astropy.io import ascii
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
import regions
import montage_wrapper
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame


## LOGGER
import logging
import logging.config
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s - %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-img','--img', dest='img', required=True, type=str, help='Input image filename (.fits)') 
	parser.add_argument('-delkeys','--delkeys', dest='delkeys', required=False, type=str, default='', help='List of FITS keywords to be removed')	
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='output.fits', help='Output filename with keywords removed') 
	
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

	inputfile= args.img
	delkeys= [str(x.strip()) for x in args.delkeys.split(',')]
	outfile= args.outfile
	
	#===========================
	#==   READ IMAGE
	#===========================
	logger.info("Read image ...")
	hdu= fits.open(inputfile)
	data= hdu[0].data
	header= hdu[0].header

	nchan = len(data.shape)
	if nchan == 4:
		data = data[0, 0, :, :]
	
	shape= data.shape	

	#===========================
	#==   REMOVE KEYWORDS
	#===========================
	for delkey in delkeys:
		if delkey in header:
			del header[delkey]

	#===========================
	#==   WRITE FILE
	#===========================
	logger.info("Write to file %s ..." % (outfile))
	hdu_out= fits.PrimaryHDU(data, header)
	hdul = fits.HDUList([hdu_out])
	hdul.writeto(outfile, overwrite=True)
	
	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

