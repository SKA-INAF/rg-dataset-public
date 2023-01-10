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
	parser.add_argument('-datadir','--datadir', dest='datadir', required=True, type=str, help='Directory containing masks & json') 
	
	# - Survey options
	parser.add_argument('-telescope','--telescope', dest='telescope', required=False, type=str, default='unknown', help='Telescope name') 
	parser.add_argument('-pixsize_x','--pixsize_x', dest='pixsize_x', required=False, type=float, default=1.0, help='Map pixel size along x (arcsec)') 
	parser.add_argument('-pixsize_y','--pixsize_y', dest='pixsize_y', required=False, type=float, default=1.0, help='Map pixel size along y (arcsec)') 

	# - Mask options
	parser.add_argument('-class_remap','--class_remap', dest='class_remap', required=False, type=str, default='',help='Class remap dictionary') 
	parser.add_argument('-mvaldict','--mvaldict', dest='mvaldict', required=False, type=str, default='',help='Mask value dictionary') 
	parser.add_argument('-npix_thr','--npix_thr', dest='npix_thr', required=False, type=int, default=5, help='Threshold in number of pixels below which source is skipped') 

	# - Output options
	parser.add_argument('-draw','--draw', dest='draw', action='store_true')	
	parser.set_defaults(draw=False)

	parser.add_argument('-save_json','--save_json', dest='save_json', action='store_true')	
	parser.set_defaults(save_json=False)

	parser.add_argument('-save_masks','--save_masks', dest='save_masks', action='store_true')	
	parser.set_defaults(save_masks=False)
	
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

	# - Input options
	datadir= args.datadir
	
	# - Survey options
	pixsize_x= args.pixsize_x
	pixsize_y= args.pixsize_y
	telescope= args.telescope

	# - Mask options
	npix_thr= args.npix_thr
	mvaldict_str= args.mvaldict
	class_remap_str= args.class_remap
	print(mvaldict_str)

	mask_vals= {}
	
	if mvaldict_str:
		mask_vals= json.loads(mvaldict_str)
	
	class_remap= {}
	if class_remap_str:
		class_remap= json.loads(class_remap_str)
	
	print("== mask vals ==")
	print(mask_vals)
	print(type(mask_vals))

	print("== class_remap ==")
	print(class_remap)
	
	# - Output file
	draw= args.draw
	save_json= args.save_json
	save_masks= args.save_masks
	

	#===========================
	#==   PROCESS JSONs
	#===========================
	filenames= [os.path.join(datadir, f) for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]

	for filename in filenames:
		if not filename.endswith(".json"):
			continue


		#===========================
		#==   PARSE JSON
		#===========================
		logger.info("Processing file %s ..." % (filename))
		filename_base= os.path.basename(filename)
		filename_base_noext= os.path.splitext(filename_base)[0]

		with open(filename, 'r') as fp:
			datadict= json.load(fp)

		#===========================
		#==   READ IMAGE
		#===========================
		inputfile= datadict['img']
		inputfile_nopath= os.path.basename(inputfile)
		inputfile_relpath= '../imgs/' + inputfile_nopath

		logger.info("Reading image %s ..." % inputfile)
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
			logger.error("Missing beam information in header!")
			return 1

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
		#==   READ MASKS
		#===========================
		# - Init data
		objs= datadict['objs']
		masks= []
		class_names= []
		is_good_mask= []

		# - Init output dictionary
		summary_info= {"img": inputfile_relpath, "objs": []}

		# - Loop over mask and compute pars
		for obj in objs:
			sname= obj['name']
			class_name_old= obj['class']
			filename_mask= obj['mask']

			# - Read mask data			
			mask= fits.open(filename_mask)[0].data
			bmap= np.copy(mask)
			bmap[bmap>0]= 1
			bmap= bmap.astype(np.uint8)

			# - Find number of islands
			label_img= measure.label(bmap)
			regprops= regionprops(label_image=label_img, intensity_image=data)
			nislands= len(regprops)
		
			# - Remap class name
			class_name= class_remap[class_name_old]
			if nislands>1:
				class_name= "extended-multicomp"
				if class_name_old!="galaxy":
					logger.error("Anomalous mask for file %s, check!" % (filename_mask))


			class_names.append(class_name)

			# - Replace mask 1s with mask value (multi-level)
			mask_value= mask_vals[class_name]
			mask[mask>0] = mask_value

			cond_mask= np.logical_and(np.isfinite(mask), mask!=0)
			npix= np.count_nonzero(mask[cond_mask])
			is_good= True
			if npix<npix_thr:
				is_good= False
				logger.warn("Setting mask %s (filename=%s) as bad ..." % (sname, filename_mask))

			is_good_mask.append(is_good)

			masks.append(mask)

		#===========================
		#==   COMPUTE NOISE
		#===========================
		noise_mask= np.ones(shape)
		for i in range(len(masks)):
			is_good= is_good_mask[i]
			if not is_good:
				continue
			mask= masks[i]
			noise_mask[mask>0]= 0
		
		cond_mask= np.logical_and(np.isfinite(noise_mask), noise_mask!=0)
		cond_img= np.isfinite(data)
		cond= np.logical_and(cond_mask, cond_img)
		noise_data_1d= data[cond]
		noise_mean, noise_median, noise_rms= sigma_clipped_stats(noise_data_1d)
	
		#===========================
		#==   PROCESS MASKS
		#===========================
		for k in range(len(masks)):
			is_good= is_good_mask[k]
			if not is_good:
				continue
			filename_mask= objs[k]['mask']
			sname= objs[k]['name']
			class_name_old= objs[k]['class']
			class_name= class_names[k]
			tag_bright= 0
			tag_flagged= 0
			if 'sidelobe-near' in objs[k]:
				tag_bright= objs[k]['sidelobe-near']
			if 'sidelobe-mixed' in objs[k]:
				tag_flagged= objs[k]['sidelobe-mixed']

			mask= masks[k]

			logger.info("Processing mask file %s ..." % (filename_mask))
			
			# - Find contours
			bmap= np.copy(mask)
			bmap[bmap>0]= 1
			bmap= bmap.astype(np.uint8)

			logger.info("Find obj %s contours ..." % (sname))
			contours= cv.findContours(bmap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			contours= imutils.grab_contours(contours)
			logger.info("#%d contours found ..." % (len(contours)))

			# - Find number of islands
			logger.info("Find region properties for obj %s ..." % (sname))
			label_img= measure.label(bmap)
			regprops= regionprops(label_image=label_img, intensity_image=data)
			logger.info("#%d regprops found ..." % (len(regprops)))
			nislands= len(regprops)
		
			logger.info("Processing source %s: class(old)=%s, class=%s, mask_value=%d" % (sname, class_name_old, class_name, mask_value))

			# - Find number of pixels 
			cond= np.logical_and(np.isfinite(mask), mask!=0)
			npix_tot= np.count_nonzero(cond)
			
			# - Find signal-to-noise
			data_1d= data[cond]
			Stot= np.nansum(data_1d)
			Sbkg= noise_median*npix_tot
			S= Stot-Sbkg
			Serr_noise= noise_rms*np.sqrt(npix_tot) # NOT SURE THIS IS CORRECT, CHECK!!!
			SNR= S/Serr_noise

			logger.info("Object %s: Stot=%f, S=%f, S_noise=%f, npix=%d, rms=%f, SNR=%f" % (sname, Stot, S, Serr_noise, npix_tot, noise_rms, SNR))
		

			# - Find bounding box of entire object (merging contours)
			#   NB: boundingRect returns Rect(x_top-left, y_top-left, width, height)
			#   NB2: patches.Rectangle wants top-left corner (bottom visually)
			#      top-left means bottom visually as y origin is at the top and is increasing from top to bottom 
			if not contours:
				logger.error("No contours found for object %s, skip it!" % (sname))
				continue

			for i in range(len(contours)):
				if i==0:
					contours_merged= contours[i]
				else:
					contours_merged= np.append(contours_merged, contours[i], axis=0)
		
			bbox= cv.boundingRect(contours_merged)
			bbox_x_tl= bbox[0] 
			bbox_y_tl= bbox[1]
			bbox_w= bbox[2] 
			bbox_h= bbox[3]
			bbox_x= bbox_x_tl + 0.5*bbox_w
			bbox_y= bbox_y_tl + 0.5*bbox_h
			
			logger.info("Bounding box for obj %s ..." % (sname))
			print(bbox)

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
		
			bbox_min_points = cv.boxPoints(bbox_min)
			#bbox_min_points = np.int0(bbox_min_points)

			logger.info("Min area bounding box for obj %s ..." % (sname))
		
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

			# - Compute other parameters
			beamAreaRatio= float(npix_tot)/float(npixInBeam)
			minSizeVSBeam= float(min(bbox_min_w,bbox_min_h))/beamWidthInPixel
			maxSizeVSBeam= float(max(bbox_min_w,bbox_min_h))/beamWidthInPixel
			minSizeVSImg= min(float(bbox_w)/float(nx), float(bbox_h)/float(ny))
			maxSizeVSImg= max(float(bbox_w)/float(nx), float(bbox_h)/float(ny))
	
			

			# - Fill data output dictionary
			d= {
				"mask": filename_mask, 
				"class": class_name, 
				"name": sname, 
				"sidelobe-near": tag_bright, 
				"sidelobe-mixed": tag_flagged,
				"npix": npix_tot,
				"nislands": nislands,
				"bbox_x": float(bbox_min_x),
				"bbox_y": float(bbox_min_y),
				"bbox_w": float(bbox_min_w),
				"bbox_h": float(bbox_min_h),
				"bbox_angle": float(bbox_min_angle),
				"Stot": float(Stot),
				"rms": float(noise_rms),
				"bkg": float(noise_median),
				"snr": float(SNR),
				"telescope": telescope,
				"bmaj": bmaj,
				"bmin": bmin,
				"nbeams": beamAreaRatio,
				"minsize_beam": minSizeVSBeam,
 				"maxsize_beam": maxSizeVSBeam,
				"minsize_img_fract": minSizeVSImg,
				"maxsize_img_fract": maxSizeVSImg
			}
			summary_info["objs"].append(d)

	
		print("summary_info")
		print(summary_info)

		# - Write to file
		#outfile_summary_fullpath= filename_base_noext + '_corr.json'
		outfile_summary_fullpath= filename_base_noext + '.json'
		if save_json:
			logger.info("Writing file %s ..." % (outfile_summary_fullpath))
			with open(outfile_summary_fullpath, 'w') as fp:
				json.dump(summary_info, fp, indent=2, sort_keys=True)

	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

