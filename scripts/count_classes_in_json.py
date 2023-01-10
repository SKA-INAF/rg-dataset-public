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

	#print("--> inputfiles")		
	#print(inputfiles)

	os.chdir(currdir)

	#===========================
	#==   COUNT CLASSES
	#===========================
	#count_dict= {
	#	"galaxy": 0,
	#	"source": 0,
	#	"sidelobe": 0
	#}
	#count_dict= {
	#	"spurious": 0,
	#	"compact": 0,
	#	"extended": 0,
	#	"extended-multicomp": 0
	#}
	count_dict= {
		"spurious": 0,
		"compact": 0,
		"extended": 0
	}

	for filename in inputfiles:
		with open(filename, 'r') as infile:
			obj_list= json.load(infile)["objs"]
			#has_galaxy= False
			has_extended= False
			for item in obj_list:	
				class_label= item['class']
				count_dict[class_label]+= 1
				#if class_label=="galaxy":
				#	has_galaxy= True
				if class_label=="extended" or class_label=="extended-multicomp":
					has_extended= True

			#if not has_galaxy:
			#	print("--> No galaxy in this image %s: " % (filename))
			if not has_extended:
				print("--> No extended sources in this image %s: " % (filename))

				

	print("== COUNTS ==")
	print("#images=%d" % (len(inputfiles)))
	print(count_dict)

	#logger.info("Merging files %s into %s ..." % (str(inputfiles),outfile))
	#merge_json_files(inputfiles, outfile, key)

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

