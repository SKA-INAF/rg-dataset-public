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
from astropy.nddata.utils import Cutout2D
import regions
import skimage.measure

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


def extract_mask_connected_components(mask):
	""" Extract mask components """
	labels, ncomponents= skimage.measure.label(mask, background=0, return_num=True, connectivity=1)
	return labels, ncomponents

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-img','--img', dest='img', required=True, type=str, help='Input image filename (.fits)') 
	parser.add_argument('-region','--region', dest='region', required=True, type=str, help='DS9 region filename (.reg)') 
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, type=int, default=-1, help='Max number of regions processed (default=-1)') 
	
	# - Cutout options
	parser.add_argument('-cutout_size','--cutout_size', dest='cutout_size', required=False, type=int, default=132, help='Cutout size in pixel (default=132)') 
	parser.add_argument('-tag','--tag', dest='tag', required=False, type=str, default='source', help='Mask tag name (default=source)') 
	
	# - Output options
	parser.add_argument('-outfileprefix','--outfileprefix', dest='outfileprefix', required=False, type=str, default='source', help='Output file prefix (default=source)') 
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='mask.fits', help='Output mask filename (.fits)') 
	parser.add_argument('-imgprefixinjson','--imgprefixinjson', dest='imgprefixinjson', required=False, type=str, default='', help='Image path file prefix in json summary file') 

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
	regionfile= args.region
	nmax= args.nmax

	# - Cutout options
	cutout_size= args.cutout_size
	tag= args.tag

	# - Output file
	outfileprefix= args.outfileprefix
	outfile= args.outfile
	imgprefixinjson= args.imgprefixinjson
	
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
	#==   READ REGIONS
	#===========================
	# - Read regions
	logger.info("Read regions ...")
	region_list= regions.read_ds9(regionfile)

	logger.info("#%d regions found ..." % len(region_list))

	# - Get region names and tags
	snames= []
	mask_values= []
	counter= 0
	for r in region_list:
		counter+= 1
		sname= r.meta['text']
		snames.append(sname)
		mask_values.append(counter)
				
	# - Merge regions with same name
	index_list= sorted(find_duplicates(snames))
	#snames_merged= []
	#regions_merged= []
	#print(type(index_list))
	#print(index_list)
	
	counter= 0

	for l in index_list:
		if not l:
			continue
		counter+= 1

		for i in range(len(l)):
			index= l[i]
			mask_values[index]= counter
			#if i==0:
			#	sname= snames[index]
			#	region= region_list[index]
			#	snames_merged.append(sname)
			#else:
			#	region_merged= region.union(region_list[index])
			#	region= region_merged

		#regions_merged.append(region)
		
	logger.info("#%d regions found if merging ..." % len(index_list))
			
	print(mask_values)
			
	#======================================
	#==   CREATE MASK IMAGE FROM REGIONS
	#======================================
	logger.info("Creating mask image ...")
	mask_img= np.zeros(shape)
	
	counter= 0
	#for region in region_list:
	for i in range(len(region_list)):
		region= region_list[i]
		bbox= region.bounding_box
		region_mask= region.to_mask(mode='subpixels')
		region_mask_data= region_mask.data
		region_mask_data[region_mask_data!=0]= mask_values[i]
		mask_img[bbox.slices]= region_mask_data
	
	#mask_img[mask_img!=0]= 1
	

	#==========================================
	#==   CREATE CUTOUTS AROUND EACH REGION
	#==========================================
	counter= 0
	
	for region in region_list:
		if nmax>0 and counter>=nmax:
			logger.info("Max number of cutouts reached (%d), exit look..." % counter)
			break
		counter+= 1
		digits= '000'
		if counter>=10 and counter<100:
			digits= '00'
		elif counter>=100 and counter<1000:
			digits= '0'
		elif counter>=1000 and counter<10000:
			digits= ''
  

		sname= region.meta['text']
		logger.info("Creating cutout around region %s ..." % (sname))
		bbox= region.bounding_box
		ixmin= bbox.ixmin
		ixmax= bbox.ixmax
		iymin= bbox.iymin
		iymax= bbox.iymax
		x= ixmin + 0.5*(ixmax-ixmin)
		y= iymin + 0.5*(iymax-iymin)

		# - Extract cutout
		cutout_mask= Cutout2D(mask_img, (x,y), (cutout_size,cutout_size), mode='partial', fill_value=0)
		cutout_img= Cutout2D(data, (x,y), (cutout_size,cutout_size), mode='partial')

		cutout_mask_data= cutout_mask.data
		cutout_img_data= cutout_img.data

		img_min = np.nanmin(cutout_img_data)
		cutout_img_data[np.isnan(cutout_img_data)] = img_min	
	
		# - Save img cutout
		#outfilename_img= 'img_' + sname + '.fits'
		outfilename_img= outfileprefix + digits + str(counter) + '.fits'

		logger.info("Write cutout image around region %s ..." % (sname))
		hdu_out= fits.PrimaryHDU(cutout_img_data, header)
		hdul = fits.HDUList([hdu_out])
		hdul.writeto(outfilename_img, overwrite=True)

		# - Extract sub masks
		component_values= np.unique(cutout_mask_data)
		component_values= component_values[component_values>0]
		ncomponents= len(component_values)

		#component_labels, ncomponents= extract_mask_connected_components(cutout_mask_data)
		logger.info("Found %d sub components in mask cutout around region %s ..." % (ncomponents,sname))
		
		outfilename_img_json= imgprefixinjson + outfilename_img
		summary_info= {"img":outfilename_img_json, "objs":[]}

		for k in range(ncomponents):	
			cutout_submask= np.zeros(cutout_mask_data.shape, dtype=cutout_mask_data.dtype)
			#cutout_submask= np.where(component_labels==k+1, [1], [0])
			cutout_submask= np.where(cutout_mask_data==component_values[k], [1], [0])
			cname= 'S' + str(k+1)

			# - Save cutout	
			outfilename_mask= 'mask_' + outfileprefix + digits + str(counter) + '_obj' + str(k+1) + '.fits'

			logger.info("Write sub mask %d to file %s ..." % (k+1,outfilename_mask))
			hdu_out= fits.PrimaryHDU(cutout_submask, header)
			hdul = fits.HDUList([hdu_out])
			hdul.writeto(outfilename_mask, overwrite=True)

			# - Fill json
			d= {"mask":outfilename_mask, "class":tag, "name":cname}
			summary_info["objs"].append(d)

		# - Save json
		outfilename_json= 'mask_' + outfileprefix + digits + str(counter) + '.json'
		with open(outfilename_json, 'w') as fp:
			json.dump(summary_info, fp,indent=2, sort_keys=True)


	#===========================
	#==   WRITE MASK IMAGE FILE
	#===========================
	logger.info("Write mask to file %s ..." % outfile)
	hdu_out= fits.PrimaryHDU(mask_img, header)
	hdul = fits.HDUList([hdu_out])
	hdul.writeto(outfile, overwrite=True)

	
	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

