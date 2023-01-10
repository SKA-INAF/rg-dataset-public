## STANDARD MODULES
import sys
import numpy as np
import os
import re
import json
from collections import defaultdict
import operator as op

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## ASTROPY MODULES
from astropy.io import fits
import regions
#from regions import read_ds9

## GRAPHICS MODULES
#import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches


## LOGGER
import logging
import logging.config
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s - %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def find_duplicates(seq):
	""" Return dict with duplicated item in list"""
	tally = defaultdict(list)
	for i,item in enumerate(seq):
		tally[item].append(i)

  #return ({key:locs} for key,locs in tally.items() if len(locs)>0)
	return (locs for key,locs in tally.items() if len(locs)>0)


###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-img','--img', dest='img', required=True, type=str, help='Input image filename (.fits)') 
	parser.add_argument('-masks','--masks', dest='masks', required=True, type=str, help='List of mask filenames (.fits) separated by commas') 
	parser.add_argument('-label','--label', dest='label', required=True, type=str, help='Label {source,sidelobe,galaxy}') 
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='mask.json', help='Output json filename (.json)') 
	
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

	# - Input filelist
	inputfile= args.img
	maskfiles = [x.strip() for x in args.masks.split(',')]
	label= args.label

	# - Output file
	outfile= args.outfile
	

	#===========================
	#==   CREATE SUMMARY JSON
	#===========================
	pwd= os.getcwd()
	is_abs_path= os.path.isabs(outfile)
	outfile_noext= os.path.splitext(outfile)[0]
	outfile_summary= outfile_noext + '.json'
	outfile_summary_fullpath= outfile_summary
	if not is_abs_path:
		outfile_summary_fullpath= os.path.join(pwd,outfile_summary)
		
	
	#===========================
	#==   WRITE SUMMARY FILE
	#===========================
	# - Fill dictionary
	inputfile_nopath= os.path.basename(inputfile)
	inputfile_relpath= '../imgs/' + inputfile_nopath

	summary_info= {"img":inputfile_relpath,"objs":[]}
	
	for i in range(len(maskfiles)):
		maskfile= maskfiles[i]
		tag= label
		name= 'S' + str(i+1)
		maskfile_nopath= os.path.basename(maskfile)
		d= {"mask": maskfile_nopath, "class": tag, "name": name}
		summary_info["objs"].append(d)

	print("summary_info")
	print(summary_info)

	# - Write to file
	with open(outfile_summary_fullpath, 'w') as fp:
		json.dump(summary_info, fp,indent=2,sort_keys=True)

	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

