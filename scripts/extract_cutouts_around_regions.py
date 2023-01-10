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
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
import regions
import montage_wrapper

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
	parser.add_argument('-morph_tags','--morph_tags', dest='morph_tags', required=False, type=str, default='POINT-LIKE,COMPACT,EXTENDED,COMPACT-EXTENDED,DIFFUSE', help='Morphological source tags ')	
	#parser.add_argument('-class_tags','--class_tags', dest='class_tags', required=False, type=str, default='UNKNOWN,MULTICLASS,GALAXY,SNR,HII,PN,LBV,WR,BUBBLE', help='Morph flags of sources ')
	parser.add_argument('-class_tags','--class_tags', dest='class_tags', required=False, type=str, default='source,galaxy,sidelobe', help='Source classification tags')

	# - Cutout options
	parser.add_argument('-cutout_class_tag','--cutout_class_tag', dest='cutout_class_tag', required=False, type=str, default="", help='Tag used to select regions to extract cutout around them') 
	parser.add_argument('-cutout_size','--cutout_size', dest='cutout_size', required=False, type=int, default=132, help='Cutout size in pixel (default=132)') 
	
	# - Output options
	parser.add_argument('-outfileprefix','--outfileprefix', dest='outfileprefix', required=False, type=str, default='source', help='Output file prefix (default=source)') 
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='mask.fits', help='Output mask filename (.fits)') 
	parser.add_argument('-imgprefixinjson','--imgprefixinjson', dest='imgprefixinjson', required=False, type=str, default='', help='Image path file prefix in json summary file') 

	args = parser.parse_args()	

	return args



#===========================
#==   READ REGIONS
#===========================
def read_regions(regionfiles):
	""" Read input regions """

	# - Read regions
	logger.info("Read source regions ...")
	regs= []
	snames= []

	for regionfile in regionfiles:
		region_list= regions.read_ds9(regionfile)
		logger.info("#%d regions found in file %s ..." % (len(region_list),regionfile))
		regs.extend(region_list)
			
	logger.info("#%d source regions read ..." % (len(regs)))

	sname_last= ''

	for region in regs:
		if 'text' not in region.meta:
			logger.error("This region has no name, please check (hint: previous region name is %s)!" % sname_last)
			sys.exit(1)
		sname= region.meta['text']
		sname_last= sname
		snames.append(sname)

	# - Create compound regions from union of regions with same name
	logger.info("Creating merged multi-island regions ...")
	source_indices= sorted(find_duplicates(snames))
	scounter= 0
	regions_merged= []

	for sindex_list in source_indices:
		if not sindex_list:
			continue
		nsources= len(sindex_list)

		if nsources==1:
			sindex= sindex_list[0]
			regions_merged.append(regs[sindex])
				
		else:
			mergedRegion= copy.deepcopy(regs[sindex_list[0]])
				
			for i in range(1,len(sindex_list)):
				tmpRegion= mergedRegion.union(regs[sindex_list[i]])
				mergedRegion= tmpRegion

			regions_merged.append(mergedRegion)

	# - Select region by tag
	regs= regions_merged

	#if select_by_tag:
	#	logger.info("Selecting regions by tag (seltag=%s) ..." % seltag)
	#	regions= []
	#	for region in regions_merged:
	#		stags= region.meta['tag']
	#		has_tag= False
	#		for tag in stags:
	#			tag_value= re.sub('[{}]','',tag)
	#			has_tag= (tag_value==seltag)
	#			if has_tag:
	#				break
	#		if has_tag:
	#			regions.append(region)

	logger.info("#%d source regions left after merging multi-islands ..." % len(regs))

		
	return regs


def get_regions_from_compound(comp_region):
	""" Get all regions from compound """

	regs= []
	r1= comp_region.region1
	r2= comp_region.region2
	is_compound_1= isinstance(r1, regions.CompoundPixelRegion) 
	is_compound_2= isinstance(r2, regions.CompoundPixelRegion) 
	is_polygon_1= isinstance(r1, regions.PolygonPixelRegion)
	is_polygon_2= isinstance(r2, regions.PolygonPixelRegion)

	# - Case 1: both simple regions
	if is_polygon_1 and is_polygon_2:
		regs.append(r1)
		regs.append(r2)

	# - Case 2: first compound, second simple
	elif is_compound_1 and is_polygon_2:
		regs.append(r2)
		rr= get_regions_from_compound(r1)
		regs.extend(rr)
	
	# - Case 3: first simple, second compound
	elif is_polygon_1 and is_compound_2:
		regs.append(r1)
		rr= get_regions_from_compound(r2)
		regs.extend(rr)

	return regs


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

	# - Source region info
	morph_tags= [str(x.strip()) for x in args.morph_tags.split(',')]
	class_tags= [str(x.strip()) for x in args.class_tags.split(',')]
	
	# - Cutout options
	cutout_size= args.cutout_size
	cutout_class_tag= args.cutout_class_tag
	

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
	logger.info("Read all regions ...")
	region_list= read_regions([regionfile])

	logger.info("#%d regions found ..." % len(region_list))


	# - Get region names and tags
	snames= []
	sclass_tags= []
	smorph_tags= []
	
	for r in region_list:
		sname= r.meta['text']
		stags= r.meta['tag']
		
		smorph= 'UNKNOWN'
		sclass= 'UNKNOWN'
		for tag in stags:
			tag_value= re.sub('[{}]','',tag)
			has_class_tag= tag_value in class_tags	
			has_morph_tag= tag_value in morph_tags
			if has_class_tag:
				sclass= tag_value
			if has_morph_tag:
				smorph= tag_value

		snames.append(sname)
		sclass_tags.append(sclass)
		smorph_tags.append(smorph)


	#==================================================
	#==   CREATE CUTOUTS AROUND EACH SELECTED REGION
	#==================================================
	#  Count how many cutout are found with desidered class tag
	counter= 0
	for i in range(len(region_list)):
		region= region_list[i]
		sname= snames[i]
		sclass= sclass_tags[i]
		smorph= smorph_tags[i]

		# - Check if this source class is selected for the cutout
		if sclass!=cutout_class_tag:
			logger.info("Skipping region %s (sclass=%s, smorph=%s) as not selected for cutout (seltag=%s)..." % (sname,sclass,smorph,cutout_class_tag))
			continue

		counter+= 1

	logger.info("Found #%d cutout with desired class tag..." % counter)


	counter= 0
	
	for i in range(len(region_list)):
		region= region_list[i]
		sname= snames[i]
		sclass= sclass_tags[i]
		smorph= smorph_tags[i]

		# - Check if this source class is selected for the cutout
		if sclass!=cutout_class_tag:
			logger.debug("Skipping region %s (sclass=%s, smorph=%s) as not selected for cutout (seltag=%s)..." % (sname,sclass,smorph,cutout_class_tag))
			continue

		# - Check if max number of cutouts is reached
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
  
		
		logger.info("Creating cutout around region %s (sclass=%s, smorph=%s) ..." % (sname,sclass,smorph))

		# - Find region center
		mask= region.to_mask(mode='center')
		bbox= mask.bbox
		is_compound= isinstance(region, regions.CompoundPixelRegion) 

		ixmin= bbox.ixmin
		ixmax= bbox.ixmax
		iymin= bbox.iymin
		iymax= bbox.iymax
		x0= ixmin + 0.5*(ixmax-ixmin)
		y0= iymin + 0.5*(iymax-iymin)
		bbox_shape= bbox.shape
		
		# - Check if this source is too large for the chosen cutout size
		too_big_for_cutout= (bbox_shape[0]>=cutout_size) or (bbox_shape[1]>=cutout_size)
		cutout_eff_size= cutout_size
		if too_big_for_cutout:
			max_bbox_size= max(bbox_shape[0],bbox_shape[1])
			cutout_eff_size= int(1.2*max_bbox_size)
			logger.warn("Source %s is too large (%d,%d) for chosen cutout size (%d), increasing cutout size to %d ..." % (sname,bbox_shape[0],bbox_shape[1],cutout_size,cutout_eff_size))
			

		# - Extract cutout
		cutout= Cutout2D(data, (x0,y0), (cutout_eff_size, cutout_eff_size), mode='partial')
		cutout_img_data= cutout.data
		cutout_img_data_shape= cutout_img_data.shape
		if cutout_img_data_shape[0]!=cutout_eff_size:
			logger.warn("Cutout x size () different from expected (%d)" % (cutout_img_data_shape[0],cutout_eff_size))
			continue	
		if cutout_img_data_shape[1]!=cutout_eff_size:
			logger.warn("Cutout y size () different from expected (%d)" % (cutout_img_data_shape[1],cutout_eff_size))
			continue
		
		img_min = np.nanmin(cutout_img_data)
		cutout_img_data[np.isnan(cutout_img_data)] = img_min
	
		xmin_cutout= cutout.xmin_original
		xmax_cutout= cutout.xmax_original
		ymin_cutout= cutout.ymin_original
		ymax_cutout= cutout.ymax_original

		# - Save img cutout
		outfilename_img= outfileprefix + digits + str(counter) + '.fits'

		logger.info("Write cutout image around region %s ..." % (sname))
		hdu_out= fits.PrimaryHDU(cutout_img_data, header)
		hdul = fits.HDUList([hdu_out])
		hdul.writeto(outfilename_img, overwrite=True)

		# - Save shrinked img cutout
		if cutout_eff_size!=cutout_size:
			logger.info("Resizing cutout to desired cutout size %d ..." % cutout_size)
			outfilename_img_shrinked= outfileprefix + digits + str(counter) + '_resized.fits'
			montage_wrapper.commands.mShrink(outfilename_img, outfilename_img_shrinked, factor=cutout_size, fixed_size=True)
				

		# - Init json file
		outfilename_img_json= imgprefixinjson + outfilename_img
		summary_info= {"img": outfilename_img_json, "objs": []}

		# - Extract object masks from all other regions
		nobjects= 0
		mask_img= np.zeros(cutout_img_data.shape,dtype=cutout_img_data.dtype)

		for j in range(len(region_list)):
			
			region_j= region_list[j]
			sname_j= snames[j]
			sclass_j= sclass_tags[j]
			smorph_j= smorph_tags[j]
			is_compound_j= isinstance(region_j, regions.CompoundPixelRegion) 
			rmask= region_j.to_mask(mode='center')
			bbox_j= rmask.bbox
			
			xmin_j= bbox_j.ixmin
			xmax_j= bbox_j.ixmax
			ymin_j= bbox_j.iymin
			ymax_j= bbox_j.iymax
			overlap_x= xmax_j>=xmin_cutout and xmax_cutout>=xmin_j
			overlap_y= ymax_j>=ymin_cutout and ymax_cutout>=ymin_j
			overlap= overlap_x and overlap_y

			if not overlap:
				logger.debug("Skip region %s as not overlapping ..." % sname_j)
				continue

			rmask.bbox.ixmin-= xmin_cutout
			rmask.bbox.ixmax-= xmin_cutout
			rmask.bbox.iymin-= ymin_cutout
			rmask.bbox.iymax-= ymin_cutout

			rcutout_maskimg= np.zeros(cutout_img_data_shape, np.dtype('>f8'))
	
			source_too_big_for_cutout= False
			if is_compound_j:
				cregs= get_regions_from_compound(region_j)				
				logger.info("Computing mask for compound region (%d regions) ..." % len(cregs))
				for creg in cregs:
					logger.info("--> creg")
					#print(creg)
					logger.info(type(creg))
					sname= creg.meta['text']
					rmask= creg.to_mask(mode='center')
					rmask.bbox.ixmin-= xmin_cutout
					rmask.bbox.ixmax-= xmin_cutout
					rmask.bbox.iymin-= ymin_cutout
					rmask.bbox.iymax-= ymin_cutout
					rcutout_maskimg_comp= rmask.to_image(cutout_img_data_shape)
					if rcutout_maskimg_comp is None:
						logger.error("rcutout_maskimg_comp is None, possibly this component is not overlapping, skip it...")
						source_too_big_for_cutout= True
						continue

					rcutout_maskimg+= rcutout_maskimg_comp
					
			else:
				logger.info("Computing mask for polygon region ...")
				rcutout_maskimg= rmask.to_image(cutout_img_data_shape)

			# - Check if None
			if rcutout_maskimg is None:
				logger.warn("rcutout_maskimg is None, skip this region ...")
				continue

			# - Check if too big for cutout
			if source_too_big_for_cutout:
				logger.warn("Source %s (compound? %d) is potentially too big for this cutout, will include only overlapping components..." % (sname_j,is_compound_j))

			rcutout_maskimg= rcutout_maskimg.astype(np.dtype('>f8'))
		
			nobjects+= 1
			cname= 'S' + str(nobjects)
				
			#	- Save object cutout	
			outfilename_mask= 'mask_' + outfileprefix + digits + str(counter) + '_obj' + str(nobjects) + '.fits'
			#outfilename_mask= 'mask_' + outfileprefix + digits + str(counter) + '_obj' + str(nobjects) + '_' + sname_j + '.fits'

			logger.info("Write obj mask %d to file %s ..." % (nobjects,outfilename_mask))
			hdu_out= fits.PrimaryHDU(rcutout_maskimg, header)
			hdul = fits.HDUList([hdu_out])
			hdul.writeto(outfilename_mask, overwrite=True)

			# - Add to global mask
			logger.debug("Adding cutout mask (size=%d x %d, type=%s) to global mask (size=%d x %d, type=%s) ..." % (rcutout_maskimg.shape[0],rcutout_maskimg.shape[1],rcutout_maskimg.dtype,mask_img.shape[0],mask_img.shape[1],mask_img.dtype))
			mask_img+= rcutout_maskimg

			# - Shrink images to chosen cutout size if cutout was larger?
			outfilename_mask_shrinked= outfilename_mask
			if cutout_eff_size!=cutout_size:
				logger.info("Resizing mask cutout to desired cutout size %d ..." % cutout_size)
				outfilename_mask_shrinked= 'mask_' + outfileprefix + digits + str(counter) + '_obj' + str(nobjects) + '_resized.fits'
				montage_wrapper.commands.mShrink(outfilename_mask, outfilename_mask_shrinked, factor=cutout_size, fixed_size=True)
				
			#	- Fill json
			d= {"mask": outfilename_mask_shrinked, "class": sclass_j, "name": cname}
			summary_info["objs"].append(d)


		# - Save json
		outfilename_json= 'mask_' + outfileprefix + digits + str(counter) + '.json'
		with open(outfilename_json, 'w') as fp:
			json.dump(summary_info, fp,indent=2, sort_keys=True)

		# - Save mask
		outfilename_gmask= 'mask_' + outfileprefix + digits + str(counter) + '.fits'
		logger.info("Write all-object mask to file %s ..." % outfilename_gmask)
		hdu_out= fits.PrimaryHDU(mask_img, header)
		hdul = fits.HDUList([hdu_out])
		hdul.writeto(outfilename_gmask, overwrite=True)

		# - Save shrinked mask
		if cutout_eff_size!=cutout_size:
			logger.info("Resizing all-object mask to desired cutout size %d ..." % cutout_size)
			outfilename_gmask_shrinked= 'mask_' + outfileprefix + digits + str(counter) + '.fits'
			montage_wrapper.commands.mShrink(outfilename_gmask, outfilename_gmask_shrinked, factor=cutout_size, fixed_size=True)

	
	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

