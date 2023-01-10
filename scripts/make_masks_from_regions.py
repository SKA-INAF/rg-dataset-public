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

## ROOT MODULES
#try:
#	import ROOT
#	from ROOT import TFile, TTree
#	from array import array
#	has_root_module= True
#except Exception as ex:
#	has_root_module= False

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


def is_rectangle_contour(vertices):
	""" Check if 4 corners form a rectangle """

	# - Check for 4 corners
	npoints= len(vertices)
	if npoints!=4:
		return False

	# - Check if rectangle
	x1= vertices[0][0]
	y1= vertices[0][1]
	x2= vertices[1][0]
	y2= vertices[1][1]
	x3= vertices[2][0]
	y3= vertices[2][1]
	x4= vertices[3][0]
	y4= vertices[3][1]
	return is_rectangle(x1, y1, x2, y2, x3, y3, x4, y4)

def is_rectangle(x1, y1, x2, y2, x3, y3, x4, y4):
	""" Check if 4 corners form a rectangle """
  
	cx= (x1+x2+x3+x4)/4.
	cy= (y1+y2+y3+y4)/4.

	dd1= (cx-x1)**2 + (cy-y1)**2
	dd2= (cx-x2)**2 + (cy-y2)**2
	dd3= (cx-x3)**2 + (cy-y3)**2
	dd4= (cx-x4)**2 + (cy-y4)**2
	isrect= (dd1==dd2 and dd1==dd3 and dd1==dd4) 

	return isrect



###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-img','--img', dest='img', required=True, type=str, help='Input image filename (.fits)') 
	parser.add_argument('-region','--region', dest='region', required=True, type=str, help='DS9 region filename (.reg)') 
	
	# - Survey options
	parser.add_argument('-telescope','--telescope', dest='telescope', required=False, type=str, default='unknown', help='Telescope name') 
	parser.add_argument('-pixsize_x','--pixsize_x', dest='pixsize_x', required=False, type=float, default=1.0, help='Map pixel size along x (arcsec)') 
	parser.add_argument('-pixsize_y','--pixsize_y', dest='pixsize_y', required=False, type=float, default=1.0, help='Map pixel size along y (arcsec)') 
	parser.add_argument('-npix_in_beam','--npix_in_beam', dest='npix_in_beam', required=False, type=float, default=5.0, help='Number of pixels in beam') 
		

	# - Mask options
	parser.add_argument('-mvaldict','--mvaldict', dest='mvaldict', required=False, type=str, default='',help='Mask value dictionary') 
	parser.add_argument('-split_masks','--split_masks', dest='split_masks', action='store_true')	
	parser.set_defaults(split_masks=False)

	# - Output options
	parser.add_argument('-draw','--draw', dest='draw', action='store_true')	
	parser.set_defaults(draw=False)

	parser.add_argument('-save_masks','--save_masks', dest='save_masks', action='store_true')	
	parser.set_defaults(save_masks=False)
	parser.add_argument('-save_json','--save_json', dest='save_json', action='store_true')	
	parser.set_defaults(save_json=False)
	#parser.add_argument('-save_root','--save_root', dest='save_root', action='store_true')	
	#parser.set_defaults(save_root=False)
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='mask.fits', help='Output mask filename (.fits)') 
	
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

	# - Survey options
	pixsize_x= args.pixsize_x
	pixsize_y= args.pixsize_y
	telescope= args.telescope
	npix_in_beam= args.npix_in_beam

	# - Mask options
	merge_objs= True
	split_masks= args.split_masks
	mvaldict_str= args.mvaldict
	print(mvaldict_str)

	mask_vals= {}
	if mvaldict_str:
		mask_vals= json.loads(mvaldict_str)
	
	print("== mask vals ==")
	print(mask_vals)
	print(type(mask_vals))
	
	# - Output file
	draw= args.draw
	save_masks= args.save_masks
	save_json= args.save_json
	#save_root= args.save_root
	outfile= args.outfile
	
	#===========================
	#==   READ IMAGE
	#===========================
	logger.info("Read image ...")
	hdu= fits.open(inputfile)
	data= hdu[0].data
	header= hdu[0].header
	shape= data.shape

	nx= shape[1]
	ny= shape[0]
	imgmaxsize= max(nx,ny)
	diag= np.sqrt(nx**2 + ny**2)

	has_coords= ('CDELT1' in header) and ('CDELT2' in header)
	if has_coords:
		dx= header['CDELT1']*3600. # in arcsec
		dy= header['CDELT2']*3600. 
	else:
		logger.warn("Setting map pix size to user (%f,%f), as CDELT1/CDELT2 missing in header ..." % (pixsize_x, pixsize_y))
		dx= pixsize_x
		dy= pixsize_y

	has_beam= ('BMAJ' in header) and ('BMIN' in header)
	if not has_beam:
		logger.warn("Missing beam information in header, setting it to %f x pix size ..." % (npix_in_beam))		
		bmaj= npix_in_beam*dx
		bmin= npix_in_beam*dy
		#return 1
	else:
		bmaj= header['BMAJ']*3600 # in arcsec
		bmin= header['BMIN']*3600 # in arcsec
	
	beamArea= np.pi*bmaj*bmin/(4*np.log(2)) # in arcsec^2
	pixelArea= np.abs(dx*dy) # in arcsec^2
	npixInBeam= beamArea/pixelArea
	beamWidth= np.sqrt(np.abs(bmaj*bmin)) # arcsec
	pixScale= np.sqrt(np.abs(dx*dy)) # arcsec
	beamWidthInPixel= int(math.ceil(beamWidth/pixScale))

	logger.info("Map info: beam(%f,%f), npixInBeam=%f, beamWidth=%f, pixScale=%f, beamWidthInPixel=%f" % (bmaj, bmin, npixInBeam, beamWidth, pixScale, beamWidthInPixel))

	
	#===========================
	#==   READ REGIONS
	#===========================
	logger.info("Read regions ...")
	region_list= regions.read_ds9(regionfile)

	logger.info("#%d regions found ..." % len(region_list))
	
	#===========================
	#==   MERGE REGIONS
	#===========================
	# - Get region names and tags
	snames= []
	stags= []
	stags_bright= []
	stags_flagged= []

	for r in region_list:
		print(type(r))
		sname= r.meta['text']
		tags= r.meta['tag']		
		stag= ''
		stag_bright= 0
		stag_flagged= 0

		for tag in tags:
			tag_plain= re.sub('[{}]','',tag)
			has_tag= tag_plain in mask_vals
			has_bright_tag= (tag_plain=='bright')
			has_flagged_tag= (tag_plain=='flagged')
			
			#print("sname")
			#print(sname)
			#print("tags")
			#print(tags)
			#print("tag_plain")
			#print(tag_plain)
			if has_tag:	
				stag= tag_plain
				#break
			if has_bright_tag:
				stag_bright= 1
			if has_flagged_tag:
				stag_flagged= 1

			
		stags.append(stag)
		stags_bright.append(stag_bright)
		stags_flagged.append(stag_flagged)
		snames.append(sname)
				
	# - Find index of duplicated region names	
	index_list= sorted(find_duplicates(snames))

	# - Merge regions with same name
	#regions_merged_list= []
	#snames_merged= []
	#stags_merged= []

	#if merge_objs:
	#	for l in index_list:
	#		if not l:
	#			continue
	#
	#		merged_region= None
	#		merged_region_meta= None
	#		for i in range(len(l)):
	#			index= l[i]
	#			if i==0:
	#				merged_region= region_list[index]
	#				merged_region_meta= region_list[index].meta
	#				sname= snames[index]
	#				stag= stags[index]
	#				snames_merged.append(sname)
	#				stags_merged.append(stag)
	#			else:
	#				merged_region_tmp= regions.CompoundPixelRegion(merged_region,region_list[index],op.or_)
	#				merged_region= merged_region_tmp
	#
	#		merged_region.meta= merged_region_meta
	#		regions_merged_list.append(merged_region)
	#else:
	#	for i in range(len(region_list)):		
	#		sname= snames[i]
	#		stag= stags[i]
	#		regions_merged_list.append(region_list[i])
	#		snames_merged.append(sname)
	#		stags_merged.append(stag)

	

	#===========================
	#==   CREATE MASKS
	#===========================
	masks_data= []
	mask_values= []

	for i in range(len(region_list)):
		region= region_list[i]
		tag= stags[i]
		sname= snames[i]
		logger.info("Region %s: tag=%s" % (sname,tag))
	
		# - Find mask value according to object tag
		mask_value= 1
		has_tag= tag in mask_vals
		if has_tag:
			mask_value= mask_vals[tag]

		mask_values.append(mask_value)
		
		# - Extract mask
		#mask= region.to_mask(mode='center')
		mask= region.to_mask(mode='subpixels') 		
		mask_img= mask.to_image(shape)
		
		# - Replace mask 1s with mask value (multi-level)
		mask_img[mask_img==1] = mask_value

		# - Append to masks
		masks_data.append(mask_img)
		

	#=====================================
	#==   MERGE MASKS (SAME REGION NAME)
	#=====================================
	masks_data_merged= []
	snames_merged= []
	stags_merged= []
	stags_bright_merged= []
	stags_flagged_merged= []
	mask_values_merged= []

	if merge_objs:
		for l in index_list:
			if not l:
				continue
	
			mask_data= None
			mask_value= 1			
			for i in range(len(l)):
				index= l[i]
				if i==0:
					mask_data= masks_data[index]
					mask_value= mask_values[index]
					sname= snames[index]
					stag= stags[index]
					#if len(l)>1:
					#	stag= "extended-multicomp"
						
					mask_value= mask_vals[stag]

					stag_bright= stags_bright[index]
					stag_flagged= stags_flagged[index]

					snames_merged.append(sname)
					stags_merged.append(stag)
					stags_bright_merged.append(stag_bright)
					stags_flagged_merged.append(stag_flagged)

				else:
					mask_data+= masks_data[index]
					
					
			mask_data[mask_data>0]= mask_value

			masks_data_merged.append(mask_data)
			mask_values_merged.append(mask_value)
	else:
		for i in range(len(masks_data)):		
			sname= snames[i]
			stag= stags[i]
			stag_bright= stags_bright[i]
			stag_flagged= stags_flagged[i]
			mask_value= mask_values[i]
			masks_data_merged.append(masks_data[i])
			snames_merged.append(sname)
			stags_merged.append(stag)
			stags_bright_merged.append(stag_bright)
			stags_flagged_merged.append(stag_flagged)
			mask_values_merged.append(mask_value)
	
	#===========================
	#==   SPLIT MASKS
	#===========================
	pwd= os.getcwd()
	masks_data_final= []
	outfilenames= []
	is_abs_path= os.path.isabs(outfile)
	outfile_noext= os.path.splitext(outfile)[0]
	outfile_summary= outfile_noext + '.json'
	outfile_fullpath= outfile
	outfile_summary_fullpath= outfile_summary
	outfile_root= outfile_noext + '.root'
	outfile_root_fullpath= outfile_root
	if not is_abs_path:
		outfile_fullpath= os.path.join(pwd,outfile)
		outfile_summary_fullpath= os.path.join(pwd,outfile_summary)
		outfile_root_fullpath= os.path.join(pwd,outfile_root)
		
	if split_masks:
		for i in range(len(masks_data_merged)):
			mask= masks_data_merged[i]
			outfilename= outfile_noext + '_obj' + str(i+1) + '.fits'
			outfilename_fullpath= outfilename
			if not is_abs_path:
				outfilename_fullpath= os.path.join(pwd,outfilename)
			masks_data_final.append(mask)
			outfilenames.append(outfilename_fullpath)
			
	else:
		mask_data= np.zeros(shape)
		for mask in masks_data_merged:
			mask_data+= mask

		masks_data_final.append(mask_data)
		outfilenames.append(outfile_fullpath)
		
	#===========================
	#==   COMPUTE NOISE
	#===========================
	noise_mask= np.ones(shape)
	for i in range(len(masks_data_final)):
		mask= masks_data_final[i]
		noise_mask[mask>0]= 0
		
	cond_mask= np.logical_and(np.isfinite(noise_mask), noise_mask!=0)
	cond_img= np.isfinite(data)
	cond= np.logical_and(cond_mask, cond_img)
	noise_data_1d= data[cond]
	noise_mean, noise_median, noise_rms= sigma_clipped_stats(noise_data_1d)

	#plt.imshow(noise_mask)
	#plt.show()

	#===========================
	#==   COMPUTE MASK PARS
	#===========================
	nislands_list= []
	npixels_list= []
	ssum_list= []	
	bkg_list= []
	rms_list= []
	snr_list= []
	bbox_list= []
	bbox_norot_list= []

	for i in range(len(masks_data_final)):
		# - Find contours
		mask= masks_data_final[i]
		bmap= np.copy(mask)
		bmap[bmap>0]= 1
		bmap= bmap.astype(np.uint8)

		logger.info("Find obj no. %d contours ..." % (i+1))
		contours= cv.findContours(bmap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		contours= imutils.grab_contours(contours)
		logger.info("#%d contours found ..." % (len(contours)))

		# - Find number of islands
		label_img= measure.label(bmap)

		logger.info("Find region properties for obj no. %d ..." % (i+1))
		regprops= regionprops(label_image=label_img, intensity_image=data)
		logger.info("#%d regprops found ..." % (len(regprops)))
		nislands= len(regprops)
		nislands_list.append(nislands)

		# - Find number of pixels 
		cond= np.logical_and(np.isfinite(mask), mask!=0)
		npix_tot= np.count_nonzero(cond)
		npixels_list.append(npix_tot)

		# - Find signal-to-noise
		data_1d= data[cond]
		Stot= np.nansum(data_1d)
		Sbkg= noise_median*npix_tot
		S= Stot-Sbkg
		Serr_noise= noise_rms*np.sqrt(npix_tot) # NOT SURE THIS IS CORRECT, CHECK!!!
		SNR= S/Serr_noise

		bkg_list.append(noise_median)
		rms_list.append(noise_rms)
		ssum_list.append(Stot)
		snr_list.append(SNR)

		logger.info("Object no. %d: Stot=%f, S=%f, S_noise=%f, npix=%d, rms=%f, SNR=%f" % (i+1, Stot, S, Serr_noise, npix_tot, noise_rms, SNR))
		

		# - Find bounding box of entire object (merging contours)
		#   NB: boundingRect returns Rect(x_top-left, y_top-left, width, height)
		#   NB2: patches.Rectangle wants top-left corner (bottom visually)
		#      top-left means bottom visually as y origin is at the top and is increasing from top to bottom 
		if not contours:
			logger.warn("No contours found for object no. %d!" % (i+1))
			continue

		for j in range(len(contours)):
			if j==0:
				contours_merged= contours[j]
			else:
				contours_merged= np.append(contours_merged, contours[j], axis=0)
		
		bbox= cv.boundingRect(contours_merged)
		bbox_x_tl= bbox[0] 
		bbox_y_tl= bbox[1]
		bbox_w= bbox[2] 
		bbox_h= bbox[3]
		bbox_x= bbox_x_tl + 0.5*bbox_w
		bbox_y= bbox_y_tl + 0.5*bbox_h
		bbox_norot_list.append( (bbox_x,bbox_y,bbox_w,bbox_h) )
		
		logger.info("Bounding box for obj no. %d ..." % (i+1))
		print(bbox)

		# - For single-island blobs, take bounding box from scikit-image as OpenCV returns it with wrong size for rectangular shapes
		#if len(regprops)==1:	
		#	bbox_v2= regprops[0].bbox
		#	logger.info("Bounding box (skimage) for obj no. %d ..." % (i+1))
		#	print(bbox_v2)
		#	bbox_x_tl= bbox_v2[1]
		#	bbox_y_tl= bbox_v2[0]
		#	bbox_w= bbox_v2[3]-bbox_v2[1]
		#	bbox_h= bbox_v2[2]-bbox_v2[0]
		#	print("bbox_x_tl")
		#	print(bbox_x_tl)
		#	print("bbox_y_tl")
		#	print(bbox_y_tl)
		#	print("bbox_w")
		#	print(bbox_w)	
		#	print("bbox_h")
		#	print(bbox_h)

		#bbox_rect= cv.rectangle(mask,(bbox_x,bbox_y),(bbox_x + bbox_w, bbox_y + bbox_h), (0,255,0), 2)
		bbox_rect = patches.Rectangle((bbox_x_tl,bbox_y_tl), bbox_w, bbox_h, linewidth=1, edgecolor='r', facecolor='none')

		# - Find rotated bounding box of entire object
		bbox_min= cv.minAreaRect(contours_merged)
		bbox_min_x= bbox_min[0][0]
		bbox_min_y= bbox_min[0][1]
		bbox_min_w= bbox_min[1][0] 
		bbox_min_h= bbox_min[1][1]
		bbox_min_angle= bbox_min[2]
		bbox_min_x_tl= bbox_min_x - 0.5*bbox_min_w
		bbox_min_y_tl= bbox_min_y - 0.5*bbox_min_h
		
		bbox_list.append( (bbox_min_x,bbox_min_y,bbox_min_w,bbox_min_h,bbox_min_angle) )

		bbox_min_points = cv.boxPoints(bbox_min)
		#bbox_min_points = np.int0(bbox_min_points)

		logger.info("Min area bounding box for obj no. %d ..." % (i+1))
		
		print(bbox_min)
		print(bbox_min_points)
		print(bbox_min_points.shape)
		print(type(bbox_min_points))

		poly = plt.Polygon(bbox_min_points, closed=True, fill=None, edgecolor='g')
		
		# - Draw mask + bounding boxes
		if draw:
			fig, ax = plt.subplots()
			ax.imshow(mask)
			ax.add_patch(bbox_rect)
			ax.add_patch(poly)
			plt.show()

	#===========================
	#==   WRITE MASK FILE
	#===========================
	if save_masks:
		for i in range(len(masks_data_final)):
			logger.info("Write mask no. %d to file %s ..." % (i+1,outfilenames[i]))
			#hdu_out= fits.PrimaryHDU(masks_data_final[i][::-1], header)
			hdu_out= fits.PrimaryHDU(masks_data_final[i], header)
			#hdu_out= fits.PrimaryHDU(np.flip(masks_data_final[i], axis=0),header)
			hdul = fits.HDUList([hdu_out])
			hdul.writeto(outfilenames[i], overwrite=True)

	#===========================
	#==   WRITE SUMMARY FILES
	#===========================
	# - Open ROOT TFile & TTree
	#if save_root and has_root_module:
	#	logger.info("Opening ROOT file %s ..." % (outfile_root_fullpath))
	#	tfile= TFile.Open(outfile_root_fullpath, "RECREATE")

	#	logger.info("Opening ROOT TTree ...")
	#	ttree= TTree("data","data")
		
	#	#gROOT.ProcessLine(
	#	#	"struct treeData_t {\
	#	#		Int_t           Category;\
  # 	#		UInt_t          Flag;\
	#	#		Char_t          Division[4];\
  #	# 		Char_t          Nation[3];\
	#	#	};" 
	#	#);
		

	# - Fill dictionary
	inputfile_nopath= os.path.basename(inputfile)
	inputfile_relpath= '../imgs/' + inputfile_nopath

	#summary_info= {"img":inputfile,"objs":[]}
	#summary_info= {"img":inputfile_nopath,"objs":[]}
	summary_info= {
		"img": inputfile_relpath, 
		"nx": int(nx),
		"ny": int(ny),
		"dx": float(dx),
		"dy": float(dy),
		"bmaj": float(bmaj),
		"bmin": float(bmin),
		"bkg": float(bkg_list[0]),
		"rms": float(rms_list[0]),
		"telescope": telescope,
		"objs": []
	}

	for i in range(len(outfilenames)):
		maskfile= outfilenames[i]
		tag= stags_merged[i]
		tag_bright= stags_bright_merged[i]
		tag_flagged= stags_flagged_merged[i]
		nislands= nislands_list[i]
		npix= npixels_list[i]
		rms= rms_list[i]
		bkg= bkg_list[i]
		Stot= ssum_list[i]
		snr= snr_list[i]
		bbox= bbox_list[i]
		bbox_x= bbox[0]
		bbox_y= bbox[1]
		bbox_w= bbox[2]
		bbox_h= bbox[3]
		bbox_angle= bbox[4]

		bbox_norot_x= bbox_norot_list[i][0]
		bbox_norot_y= bbox_norot_list[i][1]
		bbox_norot_w= bbox_norot_list[i][2]
		bbox_norot_h= bbox_norot_list[i][3]
		bbox_norot_xmin= bbox_norot_x - 0.5*bbox_norot_w
		bbox_norot_xmax= bbox_norot_x + 0.5*bbox_norot_w	
		bbox_norot_ymin= bbox_norot_y - 0.5*bbox_norot_h
		bbox_norot_ymax= bbox_norot_y + 0.5*bbox_norot_h
		at_border_x= (bbox_norot_xmin<=0) or (bbox_norot_xmax>=nx)
		at_border_y= (bbox_norot_ymin<=0) or (bbox_norot_ymax>=ny)
		at_border= (at_border_x or at_border_y)

		#print("Source %s" % (snames_merged[i]))
		#print("bbox_norot (%f,%f,%f,%f)" % (bbox_norot_x,bbox_norot_y,bbox_norot_w,bbox_norot_h))
		#print("bbox_norot xmin/xmax=%f/%f, ymin/ymax=%f/%f" % (bbox_norot_xmin,bbox_norot_xmax,bbox_norot_ymin,bbox_norot_ymax))
		#print("at border %d/%d/%d" % (at_border_x,at_border_y,at_border))

		beamAreaRatio= float(npix)/float(npixInBeam)
		minSizeVSBeam= float(min(bbox_w,bbox_h))/beamWidthInPixel
		maxSizeVSBeam= float(max(bbox_w,bbox_h))/beamWidthInPixel

		minSizeVSImg= min(float(bbox_norot_w)/float(nx), float(bbox_norot_h)/float(ny))
		maxSizeVSImg= max(float(bbox_norot_w)/float(nx), float(bbox_norot_h)/float(ny))

		name= snames_merged[i]
		maskfile_nopath= os.path.basename(maskfile)
		#d= {"mask":maskfile,"class":tag,"name":name}
		d= {
			"mask": maskfile_nopath, 
			"class": tag, 
			"name": name, 
			"sidelobe-near": tag_bright, 
			"sidelobe-mixed": tag_flagged,
			"npix": npix,
			"nislands": nislands,
			"bbox_x": float(bbox_x),
			"bbox_y": float(bbox_y),
			"bbox_w": float(bbox_w),
			"bbox_h": float(bbox_h),
			"bbox_angle": float(bbox_angle),
			"Stot": float(Stot),
			#"rms": float(rms),
			#"bkg": float(bkg),
			"snr": float(snr),
			#"telescope": telescope,
			#"bmaj": bmaj,
			#"bmin": bmin,
			"nbeams": beamAreaRatio,
			"minsize_beam": minSizeVSBeam,
 			"maxsize_beam": maxSizeVSBeam,
			"minsize_img_fract": minSizeVSImg,
			"maxsize_img_fract": maxSizeVSImg,
			"border": int(at_border)
		}
		summary_info["objs"].append(d)

	print("summary_info")
	print(summary_info)

	# - Write to file
	if save_json:
		with open(outfile_summary_fullpath, 'w') as fp:
			json.dump(summary_info, fp, indent=2, sort_keys=True)

	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

