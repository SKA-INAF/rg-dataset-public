from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging
import fnmatch
import glob

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import json

#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

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

	parser.add_argument('-rootdir','--rootdir', dest='rootdir', required=False, type=str, default='', help='Top directory used for searching') 
	parser.add_argument('-inputfiles','--inputfiles', dest='inputfiles', required=False, type=str,default='', help='Input files separated by commas') 
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='filelist_merged.json', help='Output file name with file list')
	parser.add_argument('-key','--key', dest='key', required=False, type=str, default='data', help='Key name used for merging')
	
	args = parser.parse_args()	

	return args



def merge_json_files(filenames, outfilename, key):	
	""" Merge json files """
	result = list()
	for filename in filenames:
		with open(filename, 'r') as infile:
			result.extend(json.load(infile)[key])

	outdata= {key: result}
	with open(outfilename, 'w') as fp:
		#json.dump(result, fp)
		json.dump(outdata, fp)	



	
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

	currdir= os.getcwd()

	rootdir= os.getcwd()
	if args.rootdir!="":
		rootdir= args.rootdir
	
	inputfiles= []
	if args.inputfiles!="":
		inputfiles= [str(x.strip()) for x in args.inputfiles.split(',')]
	else:
		os.chdir(rootdir)
		for file in glob.glob("*.json"):
			filename= os.path.join(rootdir, file)
			inputfiles.append(filename)

	print("--> inputfiles")		
	print(inputfiles)

	outfile= args.outfile
	key= args.key

	os.chdir(currdir)

	#===========================
	#==   LIST FILES
	#===========================
	logger.info("Merging files %s into %s ..." % (str(inputfiles),outfile))
	merge_json_files(inputfiles, outfile, key)

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

