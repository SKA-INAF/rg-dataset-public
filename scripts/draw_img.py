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
from astropy.visualization import ZScaleInterval, ContrastBiasStretch
import regions
#from regions import read_ds9

## IMAGE PROCESSING MODULES
import cv2 as cv
import imutils
from skimage import measure
from skimage.measure import regionprops
from skimage.measure import find_contours

## LOGGER
import logging
import logging.config
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s - %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

## GRAPHICS MODULES
#import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable


###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-filelist','--filelist', dest='filelist', required=False, type=str, default='', help='Filename containing list of json summary files to be read') 
	parser.add_argument('-filename','--filename', dest='filename', required=False, type=str, default='', help='json summary filename (.json) to be read. Takes priority over list.') 
	
	#- Output options
	parser.add_argument('-save','--save', dest='save', action='store_true')	
	parser.set_defaults(save=False)

	args = parser.parse_args()	

	return args


###########################
##     UTILS
###########################
def read_fits(filename, xmin=-1, xmax=-1, ymin=-1, ymax=-1, stretch=True, normalize=True, convertToRGB=True, zscale_contrasts=[0.25,0.25,0.25], to_uint8=True, stretch_biascontrast=False, contrast=1, bias=0.5):
	""" Read FITS image """

	# - Check contrasts
	zscale_contrasts_default= [0.25,0.25,0.25]
	if len(zscale_contrasts)!=3:
		logger.warn("Size of input zscale_contrasts is !=3, ignoring inputs and using default (0.25,0.25,0.25)...")
		zscale_contrasts= zscale_contrasts_default

	# - Open file
	try:
		hdu = fits.open(filename, memmap=False)
	except Exception as ex:
		errmsg = 'ERROR: Cannot read image file: ' + filename
		logger.error(errmsg)
		return None

	# - Check if tile read
	read_tile= (xmin>=0 and xmax>=0 and ymin>=0 and ymax>=0)
	if read_tile:
		if xmax<=xmin:
			logger.error("xmax must be >xmin for tile reading!")
			return None
		if ymax<=ymin:
			logger.error("ymax must be >ymin for tile reading!")
			return None

	# - Read data
	data = hdu[0].data
	data_size = np.shape(data)
	nchan = len(data.shape)
	if nchan == 4:
		if read_tile:
			output_data = data[0, 0, ymin:ymax, xmin:xmax]
		else:
			output_data = data[0, 0, :, :]
	elif nchan == 2:
		if read_tile:
			output_data = data[ymin:ymax, xmin:xmax]
		else:
			output_data = data
	else:
		errmsg = 'ERROR: Invalid/unsupported number of channels found in file ' + filename + ' (nchan=' + str(nchan) + ')!'
		hdu.close()
		logger.error(errmsg)
		return None

	# - Convert data to float 32
	output_data = output_data.astype(np.float32)

	# - Read metadata
	header = hdu[0].header

	# - Close file
	hdu.close()

	# - Replace nan values with min pix value
	img_min = np.nanmin(output_data)
	output_data[np.isnan(output_data)] = img_min
	output_data_ch1= np.copy(output_data)
	output_data_ch2= np.copy(output_data)
	output_data_ch3= np.copy(output_data)

	# - Stretch data using zscale transform?
	if stretch:
  	#data_stretched = stretch_img(output_data)
		#output_data = data_stretched
		#output_data = output_data.astype(np.float32)
		data_stretched_ch1 = stretch_img(output_data_ch1, zscale_contrasts[0])
		output_data_ch1 = data_stretched_ch1
		output_data_ch1 = output_data_ch1.astype(np.float32)

		data_stretched_ch2 = stretch_img(output_data_ch2, zscale_contrasts[1])
		output_data_ch2 = data_stretched_ch2
		output_data_ch2 = output_data_ch2.astype(np.float32)

		data_stretched_ch3 = stretch_img(output_data_ch3, zscale_contrasts[2])
		output_data_ch3 = data_stretched_ch3
		output_data_ch3 = output_data_ch3.astype(np.float32)

	# - Stretch data using bias-contrast transform
	if stretch_biascontrast:
		data_stretched_ch1 = stretch_img_biasconstrast(output_data_ch1, contrast, bias)
		output_data_ch1 = data_stretched_ch1
		output_data_ch1 = output_data_ch1.astype(np.float32)

		data_stretched_ch2 = stretch_img_biasconstrast(output_data_ch2, contrast, bias)
		output_data_ch2 = data_stretched_ch2
		output_data_ch2 = output_data_ch2.astype(np.float32)

		data_stretched_ch3 = stretch_img_biasconstrast(output_data_ch3, contrast, bias)
		output_data_ch3 = data_stretched_ch3
		output_data_ch3 = output_data_ch3.astype(np.float32)

	# - Normalize data to [0,1]?
	if normalize:
  	#data_norm = normalize_img(output_data)
		#output_data = data_norm
		#output_data = output_data.astype(np.float32)
		data_norm_ch1 = normalize_img(output_data_ch1)
		output_data_ch1 = data_norm_ch1
		output_data_ch1 = output_data_ch1.astype(np.float32)
	
		data_norm_ch2 = normalize_img(output_data_ch2)
		output_data_ch2 = data_norm_ch2
		output_data_ch2 = output_data_ch2.astype(np.float32)

		data_norm_ch3 = normalize_img(output_data_ch3)
		output_data_ch3 = data_norm_ch3
		output_data_ch3 = output_data_ch3.astype(np.float32)

	# - Convert to RGB image?
	if convertToRGB:
		if not normalize:
			data_norm_ch1 = normalize_img(output_data_ch1)
			output_data_ch1 = data_norm_ch1
			data_norm_ch2 = normalize_img(output_data_ch2)
			output_data_ch2 = data_norm_ch2
			data_norm_ch3 = normalize_img(output_data_ch3)
			output_data_ch3 = data_norm_ch3

		data_rgb= gray2rgb([output_data_ch1, output_data_ch2, output_data_ch3], to_uint8)
		output_data = data_rgb

	else:
		output_data= output_data_ch1

	return output_data, header


def stretch_img(data, contrast=0.25):
	""" Apply z-scale stretch to image """

	transform = ZScaleInterval(contrast=contrast)
	data_stretched = transform(data)

	return data_stretched

def stretch_img_biasconstrast(data, contrast=1, bias=0.5):
	""" Apply bias-contrast stretch to image """

	transform= ContrastBiasStretch(contrast=contrast, bias=bias)
	data_stretched= transform(data)

	return data_stretched

def normalize_img(data):
	""" Normalize image to (0,1) """

	data_max = np.max(data)
	data_norm = data/data_max

	return data_norm

def gray2rgb(data_float, to_uint8=True):
	""" Convert gray image data to rgb """

	# - Scale ot [0,255], convert to uint8 if required
	if to_uint8:
		data_ch1 = np.array((data_float[0]*255).round(), dtype=np.uint8)
		data_ch2 = np.array((data_float[1]*255).round(), dtype=np.uint8)
		data_ch3 = np.array((data_float[2]*255).round(), dtype=np.uint8)
	else:
		data_ch1 = np.array(data_float[0]*255, dtype=np.float32)
		data_ch2 = np.array(data_float[1]*255, dtype=np.float32)
		data_ch3 = np.array(data_float[2]*255, dtype=np.float32)

	# - Convert to 3D
	data3 = np.stack((data_ch1, data_ch2, data_ch3), axis=-1)

	return data3


def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image. """
	for c in range(3):
		image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
	return image

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

	filelist= args.filelist
	filename= args.filename
	save= args.save

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

	#print("--> inputfiles")
	#print(inputfiles)

	
	colors_per_class= {"spurious": "red", "compact": "blue", "extended": "yellow", "extended-multisland": "green", "flagged": "cyan"}
	
	colors_per_class= {
			'bkg': (0,0,0),# black
			'spurious': (1,0,0),# red
			'compact': (0,0,1),# blue
			'extended': (1,1,0),# green	
			'extended-multisland': (1,0.647,0),# orange
			'flagged': (0,0,0),# black
			#'flagged': (1,0.647,0),# orange
			#'flagged': (0.2,1,1),# cyan
		}

	#===========================
	#==   READ FILES
	#===========================
	for inputfile in inputfiles:
		if not inputfile.endswith(".json"):
			continue

		#===========================
		#==   PARSE JSON
		#===========================
		logger.info("Processing file %s ..." % (inputfile))
		inputfile_base= os.path.basename(inputfile)
		inputfile_base_noext= os.path.splitext(inputfile_base)[0]

		with open(inputfile, 'r') as fp:
			datadict= json.load(fp)

		if 'objs' not in datadict:
			logger.error("No objs keywork in dict, skip it...")
			continue

		objs= datadict['objs']
		imgfile= datadict['img']
		imgfile_base= os.path.basename(imgfile)
		imgfile_base_noext= os.path.splitext(imgfile_base)[0]

				
		#===========================
		#==   READ IMAGE
		#===========================
		image, header= read_fits(
			imgfile, 
			stretch=True, 
			zscale_contrasts=[0.25,0.25,0.25],
			normalize=False, 
			convertToRGB=True, 
			to_uint8=False
		)

		if image is None:
			logger.error("Failed to read image from file %s!" % (imgfile))
			return 1

		#print("image shape")
		#print(image.shape)
		#print("image min/max")
		#print(np.min(image))
		#print(np.max(image))


		#===========================
		#==   READ MASKS
		#===========================
		masks= []
		bboxes= []
		class_names= []

		for obj in objs:
			maskfile= obj['mask']
			is_flagged= obj['sidelobe-mixed']
			nislands= obj['nislands']
			sclass= obj['class']
			bbox_x= obj['bbox_x']
			bbox_y= obj['bbox_y']
			bbox_h= obj['bbox_h']
			bbox_w= obj['bbox_w']
			bbox_angle= obj['bbox_angle']
		
			if nislands>1 and sclass=="extended":
				sclass= 'extended-multisland'
			if is_flagged:
				sclass= 'flagged'

			mask, mask_header= read_fits(
				maskfile, 
				stretch=False, 
				zscale_contrasts=[0.25,0.25,0.25],
				normalize=True, 
				convertToRGB=False, 
				to_uint8=True
			)

			if mask is None:
				logger.error("Failed to read mask image from file %s!" % (maskfile))
				return 1

			#print("mask shape")
			#print(mask.shape)
			#print("mask min/max")
			#print(np.min(mask))
			#print(np.max(mask))

			rect= ((bbox_x,bbox_y), (bbox_w,bbox_h), bbox_angle)
			bboxes.append(rect)
			masks.append(mask)
			class_names.append(sclass)
	

		#===========================
		#==   PLOT IMAGE
		#===========================	
		# - Create axis
		logger.debug("Create axis...")
		height, width = image.shape[:2]
		#figsize=(height,width)
		figsize=(16,16)
		fig, ax = plt.subplots(1, figsize=figsize)
	
		# - Show area outside image boundaries
		logger.debug("Show area outside image boundaries...")
		#title= imgfile_base_noext
		##ax.set_ylim(height + 10, -10)
		##ax.set_xlim(-10, width + 10)
		#ax.set_ylim(height + 2, -2)
		#ax.set_xlim(-2, width + 2)
		ax.axis('off')
		#ax.set_title(title,fontsize=30)
		
		# - Draw detected objects
		if masks:
			logger.debug("Draw detected objects...")
			masked_image = image.astype(np.uint32).copy()

			for i in range(len(masks)):
				label= class_names[i]
				color = colors_per_class[label]
				bbox= bboxes[i]
		
				# - Draw Bounding box
				bbox_points = cv.boxPoints(bbox)
				bbox_poly= plt.Polygon(bbox_points, closed=True, fill=None, edgecolor=color, linestyle='--')
				ax.add_patch(bbox_poly)

				#y1, x1, y2, x2 = self.bboxes[i]
				#p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,alpha=0.7, linestyle="solid",edgecolor=color, facecolor='none')
				#ax.add_patch(p)
	
				# Label
				#caption = label
				#ax.text(x1, y1 + 8, caption, color=color, size=20, backgroundcolor="none")

				# Mask
				mask= masks[i]
				masked_image = apply_mask(masked_image, mask, color, alpha=0.1)
				#print("masked_image min/max")
				#print(np.min(masked_image))
				#print(np.max(masked_image))
	
				# Mask Polygon
				# Pad to ensure proper polygons for masks that touch image edges.
				padded_mask = np.zeros( (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
				padded_mask[1:-1, 1:-1] = mask
				contours = find_contours(padded_mask, 0.5)
				for verts in contours:
					# Subtract the padding and flip (y, x) to (x, y)
					verts = np.fliplr(verts) - 1
					p = Polygon(verts, facecolor="none", edgecolor=color)
					#ax.add_patch(p)

			im= ax.imshow(masked_image.astype(np.uint8))

			#divider = make_axes_locatable(ax)
			#cax = divider.append_axes("right", size="5%", pad=0.05)
			#plt.colorbar(im, cax=cax)	
			
			# - Save to file
			if save:
				outfile= imgfile_base_noext + '.pdf'
				logger.info("Write plot to file %s ..." % outfile)
				fig.savefig(outfile, bbox_inches='tight')	
				plt.close(fig)
			else:
				plt.show()


	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

