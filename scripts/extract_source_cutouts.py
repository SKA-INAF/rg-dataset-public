#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import sys
import numpy as np
import os
import re
import json
from collections import defaultdict
import operator as op
import copy
from distutils.version import LooseVersion
import warnings

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## ASTROPY MODULES
from astropy.io import fits
from astropy.visualization import ZScaleInterval

## SKIMAGE MODULES
import skimage.transform
from skimage.util import img_as_float
from skimage.util import img_as_float32
from skimage.util import img_as_float64
from skimage.measure import regionprops
import scipy

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

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-filelist','--filelist', dest='filelist', required=False, type=str, default='', help='Filename containing list of json summary files to be read') 
	parser.add_argument('-filename','--filename', dest='filename', required=False, type=str, default='', help='json summary filename (.json) to be read. Takes priority over list.') 
			
	# - Data selection
	parser.add_argument('-sel_classes','--sel_classes', dest='sel_classes', required=False, type=str, default='POINT-LIKE,COMPACT,EXTENDED,EXTENDED-MULTISLAND,DIFFUSE,SPURIOUS,FLAGGED', help='Selected morphological source tags, separated by commas.')
	parser.add_argument('--skip_border_sources', dest='skip_border_sources', action='store_true')
	parser.set_defaults(skip_border_sources=False)

	parser.add_argument('--apply_npix_thr', dest='apply_npix_thr', action='store_true')
	parser.set_defaults(apply_npix_thr=False)
	parser.add_argument('-npix_thr','--npix_thr', dest='npix_thr', required=False, type=int, default=5, help='Threshold applied on the source number of pixels')
		
	# - Cutout options
	parser.add_argument('-cutout_size','--cutout_size', dest='cutout_size', required=False, type=int, default=64, help='Cutout image size in pixel. (default=64)') 
	
	# - Output options
	parser.add_argument('--save', dest='save', action='store_true')	
	parser.set_defaults(save=False)
	
	# - Draw options	
	parser.add_argument('--draw', dest='draw', action='store_true')	
	parser.set_defaults(draw=False)

	args = parser.parse_args()	

	return args


###########################
##     IMAGE RESIZE
###########################
def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
	"""A wrapper for Scikit-Image resize().

		Scikit-Image generates warnings on every call to resize() if it doesn't
		receive the right parameters. The right parameters depend on the version
		of skimage. This solves the problem by using different parameters per
		version. And it provides a central place to control resizing defaults.
	"""
	if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
  	# New in 0.14: anti_aliasing. Default it to False for backward
		# compatibility with skimage 0.13.
		return skimage.transform.resize(
			image, output_shape,
			order=order, mode=mode, cval=cval, clip=clip,
			preserve_range=preserve_range, anti_aliasing=anti_aliasing,
			anti_aliasing_sigma=anti_aliasing_sigma
		)
	else:
		return skimage.transform.resize(
			image, output_shape,
			order=order, mode=mode, cval=cval, clip=clip,
			preserve_range=preserve_range
		)


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
	""" Resizes an image keeping the aspect ratio unchanged.

		Inputs:
			min_dim: if provided, resizes the image such that it's smaller dimension == min_dim
			max_dim: if provided, ensures that the image longest side doesn't exceed this value.
			min_scale: if provided, ensure that the image is scaled up by at least this percent even if min_dim doesn't require it.    
			mode: Resizing mode:
				none: No resizing. Return the image unchanged.
				square: Resize and pad with zeros to get a square image of size [max_dim, max_dim].
				pad64: Pads width and height with zeros to make them multiples of 64. If min_dim or min_scale are provided, it scales the image up before padding. max_dim is ignored.     
				crop: Picks random crops from the image. First, scales the image based on min_dim and min_scale, then picks a random crop of size min_dim x min_dim. max_dim is not used.

		Returns:
			image: the resized image
			window: (y1, x1, y2, x2). If max_dim is provided, padding might
				be inserted in the returned image. If so, this window is the
				coordinates of the image part of the full image (excluding
				the padding). The x2, y2 pixels are not included.
			scale: The scale factor used to resize the image
			padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
	"""
    
	#image= img_as_float64(image)
	#image= img_as_float(image)

	# Keep track of image dtype and return results in the same dtype
	image_dtype = image.dtype
  
	# Default window (y1, x1, y2, x2) and default scale == 1.
	h, w = image.shape[:2]
	window = (0, 0, h, w)
	scale = 1
	#padding = [(0, 0), (0, 0), (0, 0)]
	padding = [(0, 0)]
	crop = None

	if mode == "none":
		return image, window, scale, padding, crop

	# Scale?
	if min_dim:
		# Scale up but not down
		scale = max(1, min_dim / min(h, w))

	if min_scale and scale < min_scale:
		scale = min_scale

	# Does it exceed max dim?
	if max_dim and mode == "square":
		image_max = max(h, w)
		if round(image_max * scale) > max_dim:
			scale = max_dim / image_max

	# Resize image using bilinear interpolation
	if scale != 1:
		#print("DEBUG: Resizing image from size (%d,%d) to size (%d,%d) (scale=%d)" % (h,w,round(h * scale),round(w * scale),scale))
		image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

	# Need padding or cropping?
	if mode == "square":
		# Get new height and width
		h, w = image.shape[:2]
		top_pad = (max_dim - h) // 2
		bottom_pad = max_dim - h - top_pad
		left_pad = (max_dim - w) // 2
		right_pad = max_dim - w - left_pad
		#padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
		padding = [(top_pad, bottom_pad), (left_pad, right_pad)]

		image = np.pad(image, padding, mode='constant', constant_values=0)
		window = (top_pad, left_pad, h + top_pad, w + left_pad)

	elif mode == "pad64":
		h, w = image.shape[:2]
		# Both sides must be divisible by 64
		assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
		# Height
		if h % 64 > 0:
			max_h = h - (h % 64) + 64
			top_pad = (max_h - h) // 2
			bottom_pad = max_h - h - top_pad
		else:
			top_pad = bottom_pad = 0
		
		# Width
		if w % 64 > 0:
			max_w = w - (w % 64) + 64
			left_pad = (max_w - w) // 2
			right_pad = max_w - w - left_pad
		else:
			left_pad = right_pad = 0

		padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
		image = np.pad(image, padding, mode='constant', constant_values=0)
		window = (top_pad, left_pad, h + top_pad, w + left_pad)
    
	elif mode == "crop":
		# Pick a random crop
		h, w = image.shape[:2]
		y = random.randint(0, (h - min_dim))
		x = random.randint(0, (w - min_dim))
		crop = (y, x, min_dim, min_dim)
		image = image[y:y + min_dim, x:x + min_dim]
		window = (0, 0, min_dim, min_dim)
    
	else:
		raise Exception("Mode {} not supported".format(mode))
    
	return image.astype(image_dtype), window, scale, padding, crop

def resize_mask(mask, scale, padding, crop=None):
	""" Resizes a mask using the given scale and padding.
			Typically, you get the scale and padding from resize_image() to
			ensure both, the image and the mask, are resized consistently.

			scale: mask scaling factor
			padding: Padding to add to the mask in the form [(top, bottom), (left, right), (0, 0)]
	"""

	# Suppress warning from scipy 0.13.0, the output shape of zoom() is
	# calculated with round() instead of int()
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		#mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
		mask = scipy.ndimage.zoom(mask, zoom=[scale, scale], order=0)	
	if crop is not None:
		y, x, h, w = crop
		mask = mask[y:y + h, x:x + w]
	else:
		mask = np.pad(mask, padding, mode='constant', constant_values=0)

	return mask

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
	filelist= args.filelist
	filename= args.filename
	
	inputfiles= []
	if filename=="":
		if filelist=="":
			logger.error("Both input filename & filelist are empty!")
			return 1
		else:
			try:
				f = open(filelist, 'r')
			except:
				logger.error("Failed to read file %s!" % (filelist))
				return 1
			for line in f:
				#print(repr(line))
				inputfiles.append(str(line.strip()))
	else:
		inputfiles.append(filename)
	
	# - Data selection
	sel_classes= [str(x.strip()) for x in args.sel_classes.split(',')]
	skip_border_sources= args.skip_border_sources
	apply_npix_thr= args.apply_npix_thr
	npix_thr= args.npix_thr

	# - Cutout options
	cutout_size= args.cutout_size
	

	# - Output file
	save= args.save

	# - Draw options
	draw= args.draw

	#===========================
	#==   READ FILES
	#===========================
	for inputfile in inputfiles:
		if not inputfile.endswith(".json"):
			continue

		#===========================
		#==   PARSE JSON
		#===========================
		#logger.info("Processing file %s ..." % (inputfile))
		inputfile_base= os.path.basename(inputfile)
		inputfile_base_noext= os.path.splitext(inputfile_base)[0]

		with open(inputfile, 'r') as fp:
			datadict= json.load(fp)

		if 'objs' not in datadict:
			logger.error("No objs keywork in dict, skip it...")
			continue

		objs= datadict['objs']

		#===========================
		#==   READ IMAGE
		#===========================
		imgfile= datadict['img']
		imgfile_base= os.path.basename(imgfile)
		imgfile_base_noext= os.path.splitext(imgfile_base)[0]

		logger.info("Reading image %s ..." % (imgfile))
		
		hdu= fits.open(imgfile)
		data= hdu[0].data
		header= hdu[0].header

		nchan = len(data.shape)
		if nchan == 4:
			data = data[0, 0, :, :]
	
		shape= data.shape

		#===========================
		#==   PROCESS OBJECTS
		#===========================
		obj_counter= 0

		for obj in objs:
			obj_counter+= 1
			sname= obj['name']
			is_flagged= obj['sidelobe-mixed']
			nislands= obj['nislands']
			border= obj['border']
			sclass= obj['class']	
			maskfile= obj['mask']
		
			if nislands>1 and sclass=="extended":
				sclass= 'extended-multisland'
			if is_flagged:
				sclass= 'flagged'

			sclass= sclass.upper()
		
			# - Skip non selected classes
			selected= False
			for sclass_sel in sel_classes:
				if sclass==sclass_sel:
					selected= True
					break

			if not selected:
				logger.info("Skip source %s in mask file %s with class %s ..." % (sname, maskfile, sclass))
				continue

			# - Skip border sources?
			if skip_border_sources and border==1:
				logger.info("Skip source %s in mask file %s as at border ..." % (sname, maskfile))
				continue

			# - Read object mask
			mask= fits.open(maskfile)[0].data
			mask[mask!=0]= 1
			binary_mask= np.copy(mask).astype(np.uint8)

			# - Find number of pixels 
			cond= np.logical_and(np.isfinite(mask), mask!=0)
			npix= np.count_nonzero(cond)
			if apply_npix_thr and npix<npix_thr:
				logger.info("Skip source %s in mask file %s as npix=%d<thr" % (sname, maskfile, npix))
				continue

			# - Extract blobs
			#   NB: multi-island are correctly extracted as single regionprop
			logger.info("Find region properties from mask file %s (sname=%s) ..." % (maskfile, sname))
			regprops= regionprops(label_image=binary_mask, intensity_image=data)
			logger.info("#%d regprops found in mask file %s (sname=%s) ..." % (len(regprops), maskfile, sname))
			nislands= len(regprops)
			if nislands!=1:
				logger.warn("#%d objects found in mask file %s (sname=%s) when 1 is expected, skip it..." % (nislands, maskfile, sname))
				continue


			######################
			####   DEBUG #########
			#outfilename_bmask= imgfile_base_noext + '_bmask_obj' + str(obj_counter) + '.fits'
			#logger.info("Saving binary mask to file %s ..." % (outfilename_bmask))
			#hdu_out= fits.PrimaryHDU(binary_mask, header)
			#hdul = fits.HDUList([hdu_out])
			#hdul.writeto(outfilename_bmask, overwrite=True)

			#continue
			######################



			# - Extract cutout around object
			bbox= regprops[0].bbox
			cutout_data= data[bbox[0]:bbox[2], bbox[1]:bbox[3]]
			cutout_mask= binary_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
			cutout_data_masked= np.copy(cutout_data)
			cutout_data_masked[cutout_mask==0]= 0
			logger.info("Cutout data size (%d x %d) in mask file %s (sname=%s) ..." % (cutout_data.shape[0], cutout_data.shape[1], maskfile, sname))

			#print("cutout_data.dtype")
			#print(cutout_data.dtype)
			#print("cutout_data.shape")
			#print(cutout_data.shape)

			# - Resizing cutout to desired size
			try: # work for skimage<=0.15.0
				cutout_data_resized, window, scale, padding, crop= resize_image(cutout_data, min_dim=cutout_size, max_dim=cutout_size, min_scale=None, mode="square")
			except:
				cutout_data_resized, window, scale, padding, crop= resize_image(img_as_float64(cutout_data), min_dim=cutout_size, max_dim=cutout_size, min_scale=None, mode="square")
			
			#cutout_mask_resized= resize_mask(cutout_mask, scale, padding, crop)
			####cutout_data_masked_resized= resize_mask(cutout_data_masked, scale, padding, crop)
			#cutout_data_masked_resized= np.copy(cutout_data_resized)
			#cutout_data_masked_resized[cutout_mask_resized==0]= 0

			logger.info("Resized cutout data size (%d x %d) in mask file %s (sname=%s)" % (cutout_data_resized.shape[0], cutout_data_resized.shape[1], maskfile, sname))

			# - Resizing binary mask to desired size
			try: # work for skimage<=0.15.0
				cutout_mask_resized, window, scale, padding, crop= resize_image(cutout_mask, min_dim=cutout_size, max_dim=cutout_size, min_scale=None, mode="square")
			except:
				cutout_mask_resized, window, scale, padding, crop= resize_image(cutout_mask, min_dim=cutout_size, max_dim=cutout_size, min_scale=None, mode="square")
						
			logger.info("Resized cutout mask size (%d x %d) in mask file %s (sname=%s)" % (cutout_mask_resized.shape[0], cutout_mask_resized.shape[1], maskfile, sname))
			
			# - Resizing masked cutout to desired size
			try: # work for skimage<=0.15.0
				cutout_data_masked_resized, window, scale, padding, crop= resize_image(cutout_data_masked, min_dim=cutout_size, max_dim=cutout_size, min_scale=None, mode="square")
			except:
				cutout_data_masked_resized, window, scale, padding, crop= resize_image(img_as_float64(cutout_data_masked), min_dim=cutout_size, max_dim=cutout_size, min_scale=None, mode="square")
						
			logger.info("Resized cutout data masked size (%d x %d) in mask file %s (sname=%s)" % (cutout_data_masked_resized.shape[0], cutout_data_masked_resized.shape[1], maskfile, sname))

			if draw:
				zscale_transform = ZScaleInterval(contrast=0.25)
    		
				data_stretched= zscale_transform(np.copy(data))
				cutout_data_stretched= zscale_transform(np.copy(cutout_data))
				cutout_data_masked_stretched= zscale_transform(cutout_data_masked)
				cutout_data_resized_stretched= zscale_transform(cutout_data_resized)
				cutout_data_masked_resized_stretched= zscale_transform(cutout_data_masked_resized)

				fig, axs = plt.subplots(3, 3)
				#axs[0,0].imshow(data_stretched, origin='lower')
				#axs[0,1].imshow(binary_mask, origin='lower')
				#axs[1,0].imshow(cutout_data_stretched, origin='lower')
				#axs[1,1].imshow(cutout_mask, origin='lower')
				#axs[1,2].imshow(cutout_data_masked_stretched, origin='lower')
				#axs[2,0].imshow(cutout_data_resized_stretched, origin='lower')
				#axs[2,1].imshow(cutout_mask_resized, origin='lower')
				#axs[2,2].imshow(cutout_data_masked_resized_stretched, origin='lower')

				axs[0,0].imshow(data, origin='lower')
				axs[0,1].imshow(binary_mask, origin='lower')
				axs[1,0].imshow(cutout_data, origin='lower')
				axs[1,1].imshow(cutout_mask, origin='lower')
				axs[1,2].imshow(cutout_data_masked, origin='lower')
				axs[2,0].imshow(cutout_data_resized, origin='lower')
				axs[2,1].imshow(cutout_mask_resized, origin='lower')
				axs[2,2].imshow(cutout_data_masked_resized, origin='lower')
				plt.show()

			# - Save map to file
			if save:
				outfilename_img= imgfile_base_noext + '_obj' + str(obj_counter) + '.fits'
				outfilename_bmask= imgfile_base_noext + '_bmask_obj' + str(obj_counter) + '.fits'
				outfilename_masked= imgfile_base_noext + '_masked_obj' + str(obj_counter) + '.fits'
				
				logger.info("Saving resized cutout to file %s ..." % (outfilename_img))
				hdu_out= fits.PrimaryHDU(cutout_data_resized, header)
				hdul = fits.HDUList([hdu_out])
				hdul.writeto(outfilename_img, overwrite=True)

				logger.info("Saving resized binary mask cutout to file %s ..." % (outfilename_bmask))
				hdu_out= fits.PrimaryHDU(cutout_mask_resized, header)
				hdul = fits.HDUList([hdu_out])
				hdul.writeto(outfilename_bmask, overwrite=True)

				logger.info("Saving resized masked cutout to file %s ..." % (outfilename_masked))
				hdu_out= fits.PrimaryHDU(cutout_data_masked_resized, header)
				hdul = fits.HDUList([hdu_out])
				hdul.writeto(outfilename_masked, overwrite=True)


	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

