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
import csv

## ASTROPY MODULES
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
import regions
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.stats import sigma_clipped_stats

## IMAGE MODULES
import skimage.measure
from skimage.segmentation import join_segmentations

## IMAGE PROCESSING MODULES
import cv2 as cv
import imutils
from skimage import measure
from skimage.measure import regionprops 

## GRAPHICS MODULES
#import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from shapely.geometry import Polygon, Point


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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=False, type=str, default="", help='Input data (FITS)') 
	parser.add_argument('-filelist','--filelist', dest='filelist', required=False, type=str, default="", help='Input data filelist') 
	
	parser.add_argument('-regionfile_cat','--regionfile_cat', dest='regionfile_cat', required=False, type=str, default="", help='Input region file with list of regions to be used for crossmatching detected sources. Non overlapped sources are skipped.') 

	# - Algorithm options
	parser.add_argument('-seed_thr','--seed_thr', dest='seed_thr', required=False, type=float, default=5.0, help='Seed threshold')
	parser.add_argument('-merge_thr','--merge_thr', dest='merge_thr', required=False, type=float, default=3.0, help='Merge threshold')
	parser.add_argument('-dist_thr','--dist_thr', dest='dist_thr', required=False, type=float, default=10.0, help='Max distance of source centroid from image center in pixels')
	parser.add_argument('-npix_thr','--npix_thr', dest='npix_thr', required=False, type=int, default=5, help='Minimum number of island pixels')

	# - Draw options
	parser.add_argument('--draw', dest='draw', action='store_true',help='Draw plots')	
	parser.set_defaults(draw=False)

	args = parser.parse_args()	

	return args

#===========================
#==   READ MAP
#===========================
def read_data(filename, scalefact=1):
	""" Read map from file """

	logger.info("Reading file %s ..." % filename)
	hdu= fits.open(filename, ignore_missing_end=True)[0]
	data= hdu.data
	header= hdu.header
	data_shape= data.shape
	ndim= data.ndim
		
	if ndim==3:
		data= data[0,:,:]
	elif ndim==4:
		data= data[0,0,:,:]

	data*= scalefact

	# - Get WCS
	wcs= WCS(header)
		
	return data, header, wcs

#===========================
#==   READ REGIONS
#===========================
def read_regions(regionfile):
	""" Read input regions """

	# - Check args
	if regionfile=='':
		logger.error("Empty region file given!")
		return []

	# - Read and parse regions
	logger.info("Read region file %s ..." % regionfile)
	region_list= []
	try:
		region_list= regions.read_ds9(regionfile)
	except:
		logger.error("Exception caught when reading DS9 file %s!" % regionfile)
		return []

	logger.info("#%d regions found in file %s ..." % (len(region_list),regionfile))
		
	return region_list

#===========================
#==   FIND SOURCES
#===========================
def find_sources(data, seed_thr=5, merge_thr=3, sigma_clip=3, dist_thr=-1, npix_thr=5, draw=False):
	""" Find sources """
	
	# - Get data info
	data_shape= data.shape
	y_c= data_shape[0]/2.;
	x_c= data_shape[1]/2.;

	# - Compute mask
	logger.info("Computing mask ...")
	mask= np.logical_and(data!=0, np.isfinite(data))	
	data_1d= data[mask]
	
	# - Compute clipped stats
	logger.info("Computing image clipped stats ...")
	mean, median, stddev= sigma_clipped_stats(data_1d, sigma=sigma_clip)

	# - Threshold image at seed_thr
	zmap= (data-median)/stddev
	binary_map= (zmap>merge_thr).astype(np.int32)
	
	# - Compute blobs
	logger.info("Extracting blobs ...")
	label_map= skimage.measure.label(binary_map)
	regprops= skimage.measure.regionprops(label_map, data)

	nsources= len(regprops)
	logger.info("#%d sources found ..." % nsources)
	
	if draw:
		fig, ax = plt.subplots()
		##plt.imshow(label_map)
		##plt.imshow(data)
		plt.imshow(zmap)
		plt.colorbar()

	counter= 0
	regs= []
	logger.info("#%d raw sources detected ..." % (len(regprops)))

	for regprop in regprops:
		counter+= 1

		# - Check if region max is >=seed_thr
		sslice= regprop.slice
		zmask= zmap[sslice]
		zmask_1d= zmask[np.logical_and(zmask!=0, np.isfinite(zmask))]	
		zmax= zmask_1d.max()
		if zmax<seed_thr:
			logger.info("Skip source %d as zmax=%f<thr=%f" % (counter, zmax, seed_thr))
			continue

		# - Check for centroid distance from image center
		if dist_thr>0:
			try:
				centroid= regprop.weighted_centroid
			except:
				centroid= regprop.centroid_weighted
			dist= np.sqrt( (centroid[0]-x_c)**2 + (centroid[1]-y_c)**2 )
			if dist>dist_thr:
				logger.info("Skip source %d as dist=%f<thr=%f" % (counter, dist, dist_thr))
				continue

		
		# - Update binary mask and regprops
		label= regprop.label
		bmap= np.zeros_like(binary_map)
		bmap[sslice]= label_map[sslice]
		bmap[bmap!=label]= 0
		bmap[bmap==label]= 1
		bmap= bmap.astype(np.uint8)

		# - Check number of pixels
		npix= np.count_nonzero(bmap==1)
		if npix<npix_thr:
			logger.info("Skip source %d as npix=%d<%d" % (counter, npix, npix_thr))
			continue

		#===========================
		#==   EXTRACT CONTOURS
		#===========================
		logger.info("Finding source %d (zmax=%f) contours ..." % (counter, zmax))
		contours= cv.findContours(np.copy(bmap), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		contours= imutils.grab_contours(contours)
		logger.info("#%d contours found for source %d ..." % (len(contours), counter))

		if len(contours)<=0:
			logger.error("No contours found for source %d, skip it!" % (counter))
			continue
	
		if len(contours)>1:
			logger.warn(">1 contours found for source %d, skip it!" % (counter))
			continue

		#===========================
		#==   CREATE REGION
		#===========================
		logger.info("Creating DS9 regions for source %d ..." % (counter))
		sname= "S" + str(counter)	
		meta= regions.RegionMeta({"text": sname})
		regcol= "green"
		vis= regions.RegionVisual({"color": regcol})		

		for contour in contours:
			cshape= contour.shape
			contour= contour.reshape(cshape[0],cshape[2])
			
			x, y= np.split(contour, 2, axis=1)
			x= list(x.flatten())
			y= list(y.flatten())
			
			vertices = regions.PixCoord(x=x, y=y)
			reg = regions.PolygonPixelRegion(vertices=vertices, meta=meta, visual=vis)
			if draw:
				reg.plot(ax=ax, color='red', lw=1.0)

			regs.append(reg)

		# - Draw bounding box
		if draw:
			bbox= regprop.bbox
			ymin= bbox[0]
			ymax= bbox[2]
			xmin= bbox[1]
			xmax= bbox[3]
			dx= xmax-xmin-1
			dy= ymax-ymin-1
			rect = patches.Rectangle((xmin,ymin), dx, dy, linewidth=1, edgecolor='r', facecolor='none')
			##ax.add_patch(rect)

			
	#===========================
	#==   DRAW
	#===========================
	if draw:
		plt.show()

	logger.info("#%d sources detected ..." % (len(regs)))

	return regs

#==================================
#==   SELECT OVERLAPPING REGIONS
#==================================
def polyregion2shapelypoly(reg):
	""" Converting pixel polygon region to shapely polygon """

	region_vertices= [tuple(x) for x in zip(reg.vertices.x, reg.vertices.y)]
	polygon= Polygon(region_vertices)

	return polygon

def circleregion2shapelypoly(reg):
	""" Converting pixel circle region to shapely polygon """

	point = Point(reg.center.x, reg.center.y)
	circle= point.buffer(reg.radius)
	#print(type(circle))

	return circle

def select_overlapping_regions(wcs, regs, regs_cat_wcs):
	""" Select regions overlapping with input region catalogue """

	# - Convert catalogue region to pixel coordinates and to shapely
	regs_cat= []
	polygons_cat= []
	for reg_cat_wcs in regs_cat_wcs:
		try:
			reg_cat= reg_cat_wcs.to_pixel(wcs)
		except:
			logger.error("Failed to convert catalogue region to pixel coords, skip it...")
			continue

		regs_cat.append(reg_cat)	

		# - Convert to shapely
		if isinstance(reg_cat, regions.PolygonPixelRegion):
			poly_cat= polyregion2shapelypoly(reg_cat)

		elif isinstance(reg_cat, regions.CirclePixelRegion):
			poly_cat= circleregion2shapelypoly(reg_cat)
	
		else:
			logger.error("Unsupported region catalogue type !")
			return None

		polygons_cat.append(poly_cat)

	logger.info("#%d catalogue regions ..." % (len(polygons_cat)))


	# - Convert regions to shapely polygons
	polygons= []
	for reg in regs:
		if isinstance(reg, regions.PolygonPixelRegion):
			poly= polyregion2shapelypoly(reg)

		elif isinstance(reg_cat, regions.CirclePixelRegion):
			poly= circleregion2shapelypoly(reg)
	
		else:
			logger.error("Unsupported region catalogue type !")
			return None

		polygons.append(poly)

	logger.info("#%d regions ..." % (len(polygons)))


	# - Loop over polygons and retain only overlapping ones
	regs_sel= []

	for i in range(len(regs)):
		reg= regs[i]
		polygon= polygons[i]
		overlapped= False

		for j in range(len(regs_cat)):
			polygon_cat= polygons_cat[j]
			overlapped= polygon_cat.intersects(polygon)
			if overlapped:
				break

		if overlapped:
			regs_sel.append(reg)

	logger.info("#%d regions overlapping with catalogue regions ..." % (len(regs_sel)))

	return regs_sel


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

	# - Check args
	if args.inputfile=="" and args.filelist=="":
		logger.error("Input file and filelist are empty!")
		return 1

	inputfiles= []
	if args.inputfile!="":
		inputfiles.append(args.inputfile)
	else:
		with open(args.filelist) as fp:
			for line in fp:
				filename= line.rstrip()
				inputfiles.append(filename)
	
	print("--> filenames")
	print(inputfiles)

	seed_thr= args.seed_thr
	merge_thr= args.merge_thr	
	dist_thr= args.dist_thr
	draw= args.draw
	sigma_clip= 3
	regionfile_cat= args.regionfile_cat

	#===========================
	#==   READ REGIONS
	#===========================
	regs_cat= []
	if regionfile_cat:
		logger.info("Reading region file %s ..." % (regionfile_cat))
		regs_cat= read_regions(regionfile_cat)
		if not regs_cat:
			logger.error("Failed to read region file %s!" % (regionfile_cat))
			return 1

	#===========================
	#==   EXTRACT SOURCES
	#===========================
	for filename in inputfiles:
		# - Read image from file
		logger.info("Reading image from file %s ..." % (filename))
		data, header, wcs= read_data(filename)

		# - Extract sources and get DS9 regions
		regs= find_sources(data, seed_thr=seed_thr, merge_thr=merge_thr, sigma_clip=sigma_clip, dist_thr=dist_thr, draw=draw)

		# - Select overlapping regions
		if regs_cat:
			logger.info("Select regions overlapping with catalogue regions ...")
			select_overlapping_regions(wcs, regs, regs_cat)

		# - Convert regions in sky coordinates
		regs_wcs= []
		for reg in regs:
			reg_wcs= reg.to_sky(wcs)
			regs_wcs.append(reg_wcs)

		# - Save regions
		filename_base= os.path.basename(filename)
		filename_base_noext= os.path.splitext(filename_base)[0]
		regionfile= filename_base_noext + '.reg'
		logger.info("Saving region to file %s ..." % (regionfile))
		regions.write_ds9(regs, regionfile, coordsys="image")

		regionfile_wcs= filename_base_noext + '_wcs.reg'
		logger.info("Saving region to file %s ..." % (regionfile_wcs))
		regions.write_ds9(regs_wcs, regionfile_wcs)

	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
