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

## ROOT MODULES
try:
	import ROOT
	from ROOT import TCanvas, TH1D
#	from array import array
	has_root_module= True
except Exception as ex:
	has_root_module= False
	logger.error("Cannot import ROOT modules, exit!")
	sys.exit(1)

telescope_dict= {
	"vla": 1,
	"atca": 2,
	"askap": 3,
	"meerkat": 4
}

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-filelist','--filelist', dest='filelist', required=False, type=str, default='', help='Filename containing list of json summary files to be read') 
	parser.add_argument('-filename','--filename', dest='filename', required=False, type=str, default='', help='json summary filename (.json) to be read. Takes priority over list.') 
	parser.add_argument('--skip_neg_snr', dest='skip_neg_snr', action='store_true')
	parser.set_defaults(skip_neg_snr=False)

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

	filelist= args.filelist
	filename= args.filename
	skip_neg_snr= args.skip_neg_snr

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

	#===========================
	#==   INIT HISTOS
	#===========================
	myStyle= ROOT.TStyle("myStyle","myStyle")
	myStyle.SetCanvasDefH(700) 
	myStyle.SetCanvasDefW(700) 

	myStyle.SetFrameBorderMode(0)
	myStyle.SetCanvasBorderMode(0)
	myStyle.SetPadBorderMode(0)
	myStyle.SetPadColor(0)
	myStyle.SetCanvasColor(0)
	myStyle.SetTitleFillColor(0)
	myStyle.SetTitleBorderSize(1)
	myStyle.SetStatColor(0)
	myStyle.SetStatBorderSize(1)
	myStyle.SetOptTitle(0)
	myStyle.SetOptStat(0)
	myStyle.SetOptFit(1)
	myStyle.SetOptLogx(0)
	myStyle.SetOptLogy(0)
	#myStyle.SetPalette(1,0)
	myStyle.SetTitleBorderSize(0)
	#myStyle.SetTitleX(0.1f)
	#myStyle.SetTitleW(0.8f)
	myStyle.SetStatY(0.975)                
	myStyle.SetStatX(0.95)                
	myStyle.SetStatW(0.2)                
	myStyle.SetStatH(0.15)                
	myStyle.SetTitleXOffset(0.8)
	myStyle.SetTitleYOffset(1.1)
	myStyle.SetMarkerStyle(8)
	myStyle.SetMarkerSize(0.4)
	myStyle.SetFuncWidth(1)
	myStyle.SetPadTopMargin(0.1)
	myStyle.SetPadBottomMargin(0.12)
	myStyle.SetPadLeftMargin(0.15)
	myStyle.SetPadRightMargin(0.1)
	myStyle.SetTitleSize(0.06,"X")
	myStyle.SetTitleSize(0.06,"Y")
	myStyle.SetTitleSize(0.06,"Z")
	myStyle.SetTitleFont(52,"X")
	myStyle.SetTitleFont(52,"Y")
	myStyle.SetTitleFont(52,"Z")
	myStyle.SetLabelFont(42,"X")
	myStyle.SetLabelFont(42,"Y")
	myStyle.SetLabelFont(42,"Z")
	myStyle.SetErrorX(0.)
	
	ROOT.gROOT.SetStyle("myStyle")
	ROOT.gStyle.SetPadRightMargin(0.1)

	surveyHist= ROOT.TH1D("surveyHist","surveyHist",4,0.5,4.5)
	surveyHist.GetXaxis().SetBinLabel(1,"vla")
	surveyHist.GetXaxis().SetBinLabel(2,"atca")
	surveyHist.GetXaxis().SetBinLabel(3,"askap")
	surveyHist.GetXaxis().SetBinLabel(4,"meerkat")
	
	nimgs_per_tel= {
		"vla": 0,
		"atca": 0,
		"askap": 0,
		"meerkat": 0	
	}
	
	sources_per_tel_class= {
		"vla": {"spurious": 0., "compact": 0., "extended": 0., "extended-multisland": 0., "flagged": 0.},
		"atca": {"spurious": 0., "compact": 0., "extended": 0., "extended-multisland": 0., "flagged": 0.},
		"askap": {"spurious": 0., "compact": 0., "extended": 0., "extended-multisland": 0., "flagged": 0.},
		"meerkat": {"spurious": 0., "compact": 0., "extended": 0., "extended-multisland": 0., "flagged": 0.}
	}

	sources_per_class= {
		"spurious": 0.,
		"compact": 0.,
		"extended": 0.,
		"extended-multisland": 0.,
		"flagged": 0.
	}
	flagged_sources_per_class= {
		"spurious": 0.,
		"compact": 0.,
		"extended": 0.,
		"extended-multisland": 0.,
		"flagged": 0.
	}


	fluxHistos_per_tel= {
		"vla": {
			"spurious": ROOT.TH1D("fluxHisto_vla_spurious","",100,-6,4),
			"compact": ROOT.TH1D("fluxHisto_vla_compact","",100,-6,4), 
			"extended": ROOT.TH1D("fluxHisto_vla_extended","",100,-6,4),
			"extended-multisland": ROOT.TH1D("fluxHisto_vla_extended_multi","",100,-6,4),
			"flagged": ROOT.TH1D("fluxHisto_vla_flagged","",100,-6,4),
		},
		"atca": {
			"spurious": ROOT.TH1D("fluxHisto_atca_spurious","",100,-6,4),
			"compact": ROOT.TH1D("fluxHisto_atca_compact","",100,-6,4), 
			"extended": ROOT.TH1D("fluxHisto_atca_extended","",100,-6,4),
			"extended-multisland": ROOT.TH1D("fluxHisto_atca_extended_multi","",100,-6,4),
			"flagged": ROOT.TH1D("fluxHisto_atca_flagged","",100,-6,4),
		},
		"askap": {
			"spurious": ROOT.TH1D("fluxHisto_askap_spurious","",100,-6,4),
			"compact": ROOT.TH1D("fluxHisto_askap_compact","",100,-6,4), 
			"extended": ROOT.TH1D("fluxHisto_askap_extended","",100,-6,4),
			"extended-multisland": ROOT.TH1D("fluxHisto_askap_extended_multi","",100,-6,4),
			"flagged": ROOT.TH1D("fluxHisto_askap_flagged","",100,-6,4),
		},
		"meerkat": {
			"spurious": ROOT.TH1D("fluxHisto_meerkat_spurious","",100,-6,4),
			"compact": ROOT.TH1D("fluxHisto_meerkat_compact","",100,-6,4),
			"extended": ROOT.TH1D("fluxHisto_meerkat_extended","",100,-6,4),
			"extended-multisland": ROOT.TH1D("fluxHisto_meerkat_extended_multi","",100,-6,4),
			"flagged": ROOT.TH1D("fluxHisto_meerkat_flagged","",100,-6,4)
		}
	}

	fluxHistos_per_class= {
		"spurious": ROOT.TH1D("fluxHisto_spurious","",100,-6,4),
		"compact": ROOT.TH1D("fluxHisto_compact","",100,-6,4),
		"extended": ROOT.TH1D("fluxHisto_extended","",100,-6,4),
		"extended-multisland": ROOT.TH1D("fluxHisto_extended_multi","",100,-6,4),
		"flagged": ROOT.TH1D("fluxHisto_flagged","",100,-6,4)
	}

	snrHistos_per_class= {
		"spurious": ROOT.TH1D("snrHisto_spurious","",100,-2,5),
		"compact": ROOT.TH1D("snrHisto_compact","",100,-2,5),
		"extended": ROOT.TH1D("snrHisto_extended","",100,-2,5),
		"extended-multisland": ROOT.TH1D("snrHisto_extended_multi","",100,-2,5),
		"flagged": ROOT.TH1D("snrHisto_flagged","",100,-2,5)
	}

	maxSizeHistos_per_class= {
		"spurious": ROOT.TH1D("maxSizeHisto_spurious","",100,0,2),
		"compact": ROOT.TH1D("maxSizeHisto_compact","",100,0,2),
		"extended": ROOT.TH1D("maxSizeHisto_extended","",100,0,2),
		"extended-multisland": ROOT.TH1D("maxSizeHisto_extended_multi","",100,0,2),
		"flagged": ROOT.TH1D("maxSizeHisto_flagged","",100,0,2)
	}

	maxBeamSizeHistos_per_class= {
		"spurious": ROOT.TH1D("maxBeamSizeHisto_spurious","",100,0,100),
		"compact": ROOT.TH1D("maxBeamSizeHisto_compact","",100,0,100),
		"extended": ROOT.TH1D("maxBeamSizeHisto_extended","",100,0,100),
		"extended-multisland": ROOT.TH1D("maxBeamSizeHisto_extended_multi","",100,0,100),
		"flagged": ROOT.TH1D("maxBeamSizeHisto_flagged","",100,0,100)
	}

	aspectRatioHistos_per_class= {
		"spurious": ROOT.TH1D("aspectRatioHisto_spurious","",30,0,1.5),
		"compact": ROOT.TH1D("aspectRatioHisto_compact","",30,0,1.5),
		"extended": ROOT.TH1D("aspectRatioHisto_extended","",30,0,1.5),
		"extended-multisland": ROOT.TH1D("aspectRatioHisto_extended_multi","",30,0,1.5),
		"flagged": ROOT.TH1D("aspectRatioHisto_flagged","",30,0,1.5)
	}

	snrVSaspectRatioGraph= ROOT.TGraph()
	snrVSaspectRatioGraphs_per_class= {
		"spurious": ROOT.TGraph(),
		"compact": ROOT.TGraph(),
		"extended": ROOT.TGraph(),
		"extended-multisland": ROOT.TGraph(),
		"flagged": ROOT.TGraph()
	}

	snrVSmaxSizeGraph= ROOT.TGraph()
	snrVSmaxSizeGraphs_per_class= {
		"spurious": ROOT.TGraph(),
		"compact": ROOT.TGraph(),
		"extended": ROOT.TGraph(),
		"extended-multisland": ROOT.TGraph(),
		"flagged": ROOT.TGraph()
	}

	maxSizeVSaspectRatioGraph= ROOT.TGraph()
	maxSizeVSaspectRatioGraphs_per_class= {
		"spurious": ROOT.TGraph(),
		"compact": ROOT.TGraph(),
		"extended": ROOT.TGraph(),
		"extended-multisland": ROOT.TGraph(),
		"flagged": ROOT.TGraph()
	}

	maxImgSizeVSbboxDimRatioGraph= ROOT.TGraph()
	maxImgSizeVSbboxDimRatioGraphs_per_class= {
		"spurious": ROOT.TGraph(),
		"compact": ROOT.TGraph(),
		"extended": ROOT.TGraph(),
		"extended-multisland": ROOT.TGraph(),
		"flagged": ROOT.TGraph()
	}

	maxImgSizeVSbboxDimInvRatioGraph= ROOT.TGraph()
	maxImgSizeVSbboxDimInvRatioGraphs_per_class= {
		"spurious": ROOT.TGraph(),
		"compact": ROOT.TGraph(),
		"extended": ROOT.TGraph(),
		"extended-multisland": ROOT.TGraph(),
		"flagged": ROOT.TGraph()
	}

	maxImgSizeVSminImgSizeGraph= ROOT.TGraph()
	maxImgSizeVSminImgSizeGraphs_per_class= {
		"spurious": ROOT.TGraph(),
		"compact": ROOT.TGraph(),
		"extended": ROOT.TGraph(),
		"extended-multisland": ROOT.TGraph(),
		"flagged": ROOT.TGraph()
	}

	bboxWidthVSbboxHeightGraph= ROOT.TGraph()
	bboxWidthVSbboxHeightGraphs_per_class= {
		"spurious": ROOT.TGraph(),
		"compact": ROOT.TGraph(),
		"extended": ROOT.TGraph(),
		"extended-multisland": ROOT.TGraph(),
		"flagged": ROOT.TGraph()
	}

	maxImgSizeHisto= ROOT.TH1D("maxImgSizeHisto","",50,-0.1,1.1)
	maxImgSizeHistos_per_class= {
		"spurious": ROOT.TH1D("maxImgSizeHisto_spurious","",50,-0.1,1.1),
		"compact": ROOT.TH1D("maxImgSizeHisto_compact","",50,-0.1,1.1),
		"extended": ROOT.TH1D("maxImgSizeHisto_extended","",50,-0.1,1.1),
		"extended-multisland": ROOT.TH1D("maxImgSizeHisto_extended_multi","",50,-0.1,1.1),
		"flagged": ROOT.TH1D("maxImgSizeHisto_flagged","",50,-0.1,1.1)
	}

	minImgSizeHisto= ROOT.TH1D("minImgSizeHisto","",50,-0.1,1.1)
	minImgSizeHistos_per_class= {
		"spurious": ROOT.TH1D("minImgSizeHisto_spurious","",50,-0.1,1.1),
		"compact": ROOT.TH1D("minImgSizeHisto_compact","",50,-0.1,1.1),
		"extended": ROOT.TH1D("minImgSizeHisto_extended","",50,-0.1,1.1),
		"extended-multisland": ROOT.TH1D("minImgSizeHisto_extended_multi","",50,-0.1,1.1),
		"flagged": ROOT.TH1D("minImgSizeHisto_flagged","",50,-0.1,1.1)
	}

	bboxDimRatioHisto= ROOT.TH1D("bboxDimRatioHisto","",50,0.5,10)
	bboxDimRatioHistos_per_class= {
		"spurious": ROOT.TH1D("bboxDimRatioHisto_spurious","",50,0.5,10),
		"compact": ROOT.TH1D("bboxDimRatioHisto_compact","",50,0.5,10),
		"extended": ROOT.TH1D("bboxDimRatioHisto_extended","",50,0.5,10),
		"extended-multisland": ROOT.TH1D("bboxDimRatioHisto_extended_multi","",50,0.5,10),
		"flagged": ROOT.TH1D("bboxDimRatioHisto_flagged","",50,0.5,10)
	}

	dummyGraphs_per_class= {
		"spurious": ROOT.TGraph(),
		"compact": ROOT.TGraph(),
		"extended": ROOT.TGraph(),
		"extended-multisland": ROOT.TGraph(),
		"flagged": ROOT.TGraph()
	}

	nislands_per_tel_class= {
		"vla": {
			"spurious": [0,0,0,0],
			"compact": [0,0,0,0], 
			"extended": [0,0,0,0],
			"extended-multisland": [0,0,0,0],
			"flagged": [0,0,0,0]
		},
		"atca": {
			"spurious": [0,0,0,0],
			"compact": [0,0,0,0],
			"extended": [0,0,0,0],
			"extended-multisland": [0,0,0,0],
			"flagged": [0,0,0,0]
		},
		"askap": {
			"spurious": [0,0,0,0],
			"compact": [0,0,0,0],
			"extended": [0,0,0,0],
			"extended-multisland": [0,0,0,0],
			"flagged": [0,0,0,0]
		},
		"meerkat": {
			"spurious": [0,0,0,0],
			"compact": [0,0,0,0],
			"extended": [0,0,0,0],
			"extended-multisland": [0,0,0,0],
			"flagged": [0,0,0,0]
		}
	}

	nislands_per_class= {
		"spurious": [0,0,0,0],
		"compact": [0,0,0,0], 
		"extended": [0,0,0,0],
		"extended-multisland": [0,0,0,0],
		"flagged": [0,0,0,0]
	}

	
	colors_per_class= {"spurious": ROOT.kRed, "compact": ROOT.kBlue, "extended": ROOT.kGreen+1, "extended-multisland": ROOT.kOrange, "flagged": ROOT.kBlack}
	markers_per_class= {"spurious": 21, "compact": 8, "extended": 22, "extended-multisland": 23, "flagged": 3}
	marker_sizes_per_class= {"spurious": 0.5, "compact": 0.3, "extended": 0.5, "extended-multisland": 0.5, "flagged": 0.5}
	sigma_reso= np.sqrt(2)/np.sqrt(12)
	print("sigma_reso=%f" % (sigma_reso))		


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
		tel= datadict['telescope']
		tel_int= telescope_dict[tel]
		bmaj= datadict['bmaj']
		bmin= datadict['bmin']
		dx= datadict['dx']
		dy= datadict['dy']
		pixScale= np.sqrt(np.abs(dx*dy)) # arcsec
		beamWidth= np.sqrt(np.abs(bmaj*bmin)) # arcsec	
		beamWidthInPixel= int(math.ceil(beamWidth/pixScale))

		nimgs_per_tel[tel]+= 1

		# - Fill histos
		surveyHist.Fill(tel_int)
		
		# - Loop over objects
		for obj in objs:
			sname= obj['name']
			is_flagged= obj['sidelobe-mixed']
			nislands= obj['nislands']
			sclass= obj['class']
		
			if nislands>1 and sclass=="extended":
				sclass= 'extended-multisland'
			if is_flagged:
				sclass= 'flagged'

			S= obj['Stot']
			SNR= obj['snr']

			if SNR<1:
				if skip_neg_snr:
					logger.warn("Image %s: sname=%s, SNR=%f, S=%f, skipping it ..." % (inputfile, sname, SNR, S))
					continue
				else:
					logger.warn("Image %s: sname=%s, SNR=%f, S=%f" % (inputfile, sname, SNR, S))
				

			if S<=0:
				logger.warn("Object with neg flux (Image %s: sname=%s, SNR=%f, S=%f) ..." % (inputfile, sname, SNR, S))
				#continue
			
			maxSize= obj['maxsize_img_fract']
			maxBeamSize= obj['maxsize_beam']
			minBeamSize= obj['minsize_beam']
			minsize_img_fract= obj['minsize_img_fract']
			maxsize_img_fract= obj['maxsize_img_fract']
			bbox_w= obj['bbox_w']
			bbox_h= obj['bbox_h']
			
			r1= ROOT.gRandom.Gaus(0,sigma_reso)
			r2= ROOT.gRandom.Gaus(0,sigma_reso)
			if r1>2:
				r1= 2
			if r1<-2:
				r1= -2
			if r2>2:
				r2= 2
			if r2<-2:
				r2= -2
			bbox_w_rand= max(bbox_w + r1, 0)
			bbox_h_rand= max(bbox_h + r2, 0)
			if bbox_w_rand<=0:
				bbox_w_rand= bbox_w
			if bbox_h_rand<=0:
				bbox_h_rand= bbox_h

			#print("r1=%f, r2=%f, bbox_w_rand=%f, bbox_h_rand=%f" % (r1,r2,bbox_w_rand,bbox_h_rand))
			
			minBeamSize_rand= float(min(bbox_w_rand,bbox_h_rand))/beamWidthInPixel
			maxBeamSize_rand= float(max(bbox_w_rand,bbox_h_rand))/beamWidthInPixel
			
			aspectRatio= float(maxBeamSize)/float(minBeamSize)
			aspectRatio_rand= maxBeamSize_rand/minBeamSize_rand

			bboxDimRatio= float(maxsize_img_fract)/float(minsize_img_fract)
			bboxDimInvRatio= float(minsize_img_fract)/float(maxsize_img_fract)

			if minsize_img_fract<=0:
				logger.info("Negative min_ig_fract: %s" % (inputfile))

			# - Override aspect ratio with randomized versions
			aspectRatio= aspectRatio_rand
			maxBeamSize= maxBeamSize_rand

			lgS= np.log10(S)
			lgSNR= np.log10(SNR)
			lgAspectRatio= np.log10(aspectRatio)
			lgMaxBeamSize= np.log10(maxBeamSize)

			nsources= 1.
			#if nislands>1:
			#	nsources= 1./nislands

			sources_per_tel_class[tel][sclass]+= nsources
			#nislands_per_tel_class[tel][sclass][nislands-1]+= nsources
			#nislands_per_class[sclass][nislands-1]+= nsources
			if nislands>3:
				nislands_per_tel_class[tel][sclass][3]+= nsources
				nislands_per_class[sclass][3]+= nsources
			else:
				nislands_per_tel_class[tel][sclass][nislands-1]+= nsources
				nislands_per_class[sclass][nislands-1]+= nsources

			sources_per_class[sclass]+= nsources
			if is_flagged:
				flagged_sources_per_class[sclass]+= nsources

			fluxHistos_per_class[sclass].Fill(lgS)
			snrHistos_per_class[sclass].Fill(lgSNR)
			maxBeamSizeHistos_per_class[sclass].Fill(maxBeamSize)
			maxSizeHistos_per_class[sclass].Fill(maxSize)
			aspectRatioHistos_per_class[sclass].Fill(lgAspectRatio)

			snrVSaspectRatioGraph.SetPoint(snrVSaspectRatioGraph.GetN(), lgSNR, lgAspectRatio)
			snrVSaspectRatioGraphs_per_class[sclass].SetPoint(snrVSaspectRatioGraphs_per_class[sclass].GetN(), lgSNR, lgAspectRatio)
			snrVSmaxSizeGraph.SetPoint(snrVSmaxSizeGraph.GetN(), lgSNR, lgMaxBeamSize)
			snrVSmaxSizeGraphs_per_class[sclass].SetPoint(snrVSmaxSizeGraphs_per_class[sclass].GetN(), lgSNR, lgMaxBeamSize)

			maxSizeVSaspectRatioGraph.SetPoint(maxSizeVSaspectRatioGraph.GetN(), lgSNR, lgMaxBeamSize)
			maxSizeVSaspectRatioGraphs_per_class[sclass].SetPoint(maxSizeVSaspectRatioGraphs_per_class[sclass].GetN(), lgMaxBeamSize, lgAspectRatio)
		
			maxImgSizeVSbboxDimRatioGraph.SetPoint(maxImgSizeVSbboxDimRatioGraph.GetN(), maxsize_img_fract, bboxDimRatio)
			maxImgSizeVSbboxDimRatioGraphs_per_class[sclass].SetPoint(maxImgSizeVSbboxDimRatioGraphs_per_class[sclass].GetN(), maxsize_img_fract, bboxDimRatio)
	
			maxImgSizeVSbboxDimInvRatioGraph.SetPoint(maxImgSizeVSbboxDimInvRatioGraph.GetN(), maxsize_img_fract, bboxDimInvRatio)
			maxImgSizeVSbboxDimInvRatioGraphs_per_class[sclass].SetPoint(maxImgSizeVSbboxDimInvRatioGraphs_per_class[sclass].GetN(), maxsize_img_fract, bboxDimInvRatio)

			maxImgSizeVSminImgSizeGraph.SetPoint(maxImgSizeVSminImgSizeGraph.GetN(), minsize_img_fract, maxsize_img_fract)
			maxImgSizeVSminImgSizeGraphs_per_class[sclass].SetPoint(maxImgSizeVSminImgSizeGraphs_per_class[sclass].GetN(), minsize_img_fract, maxsize_img_fract)

			bboxWidthVSbboxHeightGraph.SetPoint(bboxWidthVSbboxHeightGraph.GetN(), bbox_w, bbox_h)
			bboxWidthVSbboxHeightGraphs_per_class[sclass].SetPoint(bboxWidthVSbboxHeightGraphs_per_class[sclass].GetN(), bbox_w, bbox_h)

			maxImgSizeHisto.Fill(maxsize_img_fract)
			maxImgSizeHistos_per_class[sclass].Fill(maxsize_img_fract)
			minImgSizeHisto.Fill(minsize_img_fract)
			minImgSizeHistos_per_class[sclass].Fill(minsize_img_fract)
			bboxDimRatioHisto.Fill(bboxDimRatio)
			bboxDimRatioHistos_per_class[sclass].Fill(bboxDimRatio);


	#===========================
	#==   COMPUTE STATS
	#===========================
	nimgs= sum(nimgs_per_tel.values())
	nimgs_fract_per_tel = {key: value / nimgs for key, value in nimgs_per_tel.items()}

	print("== nimgs ==")
	print("nimgs=%d" % (nimgs))
	print(nimgs_per_tel)
	print(nimgs_fract_per_tel)

	print("")

	print("== sclass ==")
	print(sources_per_class)
	print(sources_per_tel_class)

	print("")

	print("== sclass (flagged) ==")
	print(flagged_sources_per_class)

	print("")

	print("nislands")
	print(nislands_per_class)
	print(nislands_per_tel_class)

	print("")

	#===========================
	#==   PLOT
	#===========================
	# - Draw tel stats
	logger.info("Plotting telescope stats ...")
	plot_telstats= ROOT.TCanvas("plot_telstats","plot_telstats")
	plot_telstats.cd()
	
	surveyHist.SetLineColor(ROOT.kBlack)
	surveyHist.SetFillStyle(1001)
	surveyHist.SetFillColor(ROOT.kAzure)
	surveyHist.Draw("hist")
	
	# - Draw flux plot
	logger.info("Plotting flux stats ...")
	plot_flux= ROOT.TCanvas("plot_flux","plot_flux")
	plot_flux.cd()

	plotBkg_flux= ROOT.TH2D("plotBkg_flux","",100,-6,4,100,0,1)
	plotBkg_flux.SetXTitle("log_{10}(S/Jy)")
	plotBkg_flux.SetYTitle("#sources")
	plotBkg_flux.Draw()

	for key in fluxHistos_per_class:
		fluxHistos_per_class[key].SetLineColor(colors_per_class[key])
		fluxHistos_per_class[key].DrawNormalized("hist same")
	
	# - Draw SNR plot
	logger.info("Plotting SNR stats ...")
	plot_snr= ROOT.TCanvas("plot_snr","plot_snr")
	plot_snr.cd()

	plotBkg_snr= ROOT.TH2D("plotBkg_snr","",100,-2,5,100,0,1)
	plotBkg_snr.SetXTitle("log_{10}(SNR)")
	plotBkg_snr.SetYTitle("#sources")
	plotBkg_snr.Draw()

	for key in snrHistos_per_class:
		snrHistos_per_class[key].SetLineColor(colors_per_class[key])
		snrHistos_per_class[key].DrawNormalized("hist same")

	
	# - Draw aspect ratio plot
	logger.info("Plotting aspect ratio stats ...")
	plot_aspectRatio= ROOT.TCanvas("plot_aspectRatio","plot_aspectRatio")
	plot_aspectRatio.cd()

	plotBkg_aspectRatio= ROOT.TH2D("plotBkg_aspectRatio","",100,-0.5,1.5,100,0,0.3)
	plotBkg_aspectRatio.SetXTitle("log_{10}(maxSize/minSize)")
	plotBkg_aspectRatio.SetYTitle("#sources")
	plotBkg_aspectRatio.Draw()

	for key in aspectRatioHistos_per_class:
		aspectRatioHistos_per_class[key].SetLineColor(colors_per_class[key])
		aspectRatioHistos_per_class[key].DrawNormalized("hist same")


	# - Draw max size wrt img size plot
	logger.info("Plotting max size wrt img size ...")
	plot_maxImgSize= ROOT.TCanvas("plot_maxImgSize","plot_maxImgSize")
	plot_maxImgSize.cd()

	plotBkg_maxImgSize= ROOT.TH2D("plotBkg_maxImgSize","",100,-0.1,1.1,100,0,1)
	plotBkg_maxImgSize.SetXTitle("maxSize/imgSize")
	plotBkg_maxImgSize.SetYTitle("#sources")
	plotBkg_maxImgSize.Draw()

	maxImgSizeHisto.SetLineColor(ROOT.kBlack)
	maxImgSizeHisto.SetLineWidth(2)
	maxImgSizeHisto.DrawNormalized("hist same")

	for key in maxImgSizeHistos_per_class:
		maxImgSizeHistos_per_class[key].SetLineColor(colors_per_class[key])
		maxImgSizeHistos_per_class[key].DrawNormalized("hist same")

	# - Draw min size wrt img size plot
	logger.info("Plotting min size wrt img size ...")
	plot_minImgSize= ROOT.TCanvas("plot_minImgSize","plot_minImgSize")
	plot_minImgSize.cd()

	plotBkg_minImgSize= ROOT.TH2D("plotBkg_minImgSize","",100,-0.1,1.1,100,0,1)
	plotBkg_minImgSize.SetXTitle("minSize/imgSize")
	plotBkg_minImgSize.SetYTitle("#sources")
	plotBkg_minImgSize.Draw()

	minImgSizeHisto.SetLineColor(ROOT.kBlack)
	minImgSizeHisto.SetLineWidth(2)
	minImgSizeHisto.DrawNormalized("hist same")

	for key in minImgSizeHistos_per_class:
		minImgSizeHistos_per_class[key].SetLineColor(colors_per_class[key])
		minImgSizeHistos_per_class[key].DrawNormalized("hist same")

	# - Draw bbox dim ratio plot
	logger.info("Plotting bbox dim ratio ...")
	plot_bboxDimRatio= ROOT.TCanvas("plot_bboxDimRatio","plot_bboxDimRatio")
	plot_bboxDimRatio.cd()

	plotBkg_bboxDimRatio= ROOT.TH2D("plotBkg_bboxDimRatio","",100,0.5,10,100,0,1)
	plotBkg_bboxDimRatio.SetXTitle("maxBBoxSize/minBBoxSize")
	plotBkg_bboxDimRatio.SetYTitle("#sources")
	plotBkg_bboxDimRatio.Draw()

	bboxDimRatioHisto.SetLineColor(ROOT.kBlack)
	bboxDimRatioHisto.SetLineWidth(2)
	bboxDimRatioHisto.DrawNormalized("hist same")

	for key in bboxDimRatioHistos_per_class:
		bboxDimRatioHistos_per_class[key].SetLineColor(colors_per_class[key])
		bboxDimRatioHistos_per_class[key].DrawNormalized("hist same")

	
	# - Draw aspect ratio vs SNR
	logger.info("Plotting aspect ratio VS SNR ...")
	plot_aspectRatioVSSNR= ROOT.TCanvas("plot_aspectRatioVSSNR","plot_aspectRatioVSSNR")
	plot_aspectRatioVSSNR.cd()

	plotBkg_aspectRatioVSSNR= ROOT.TH2D("plotBkg_aspectRatioVSSNR","",100,-2,5,100,-0.5,2)
	plotBkg_aspectRatioVSSNR.GetXaxis().SetTitle("log_{10}(SNR)")
	plotBkg_aspectRatioVSSNR.GetYaxis().SetTitle("log_{10}(maxSize/minSize)")
	plotBkg_aspectRatioVSSNR.Draw()
	
	snrVSaspectRatioGraph.SetMarkerSize(0.1)
	snrVSaspectRatioGraph.SetMarkerColor(ROOT.kWhite)
	snrVSaspectRatioGraph.GetXaxis().SetTitle("log_{10}(SNR)")
	snrVSaspectRatioGraph.GetYaxis().SetTitle("log_{10}(maxSize/minSize)")
	#snrVSaspectRatioGraph.Draw("AP")

	legend_aspectRatioVSSNR= ROOT.TLegend(0.7,0.7,0.9,0.9)
	legend_aspectRatioVSSNR.SetFillColor(0)
	legend_aspectRatioVSSNR.SetTextFont(52)
	legend_aspectRatioVSSNR.SetTextSize(0.04)

	for key in snrVSaspectRatioGraphs_per_class:
		snrVSaspectRatioGraphs_per_class[key].SetMarkerSize(marker_sizes_per_class[key])
		snrVSaspectRatioGraphs_per_class[key].SetMarkerStyle(markers_per_class[key])
		snrVSaspectRatioGraphs_per_class[key].SetMarkerColor(colors_per_class[key])
		snrVSaspectRatioGraphs_per_class[key].Draw("P")
	
		dummyGraphs_per_class[key].SetMarkerSize(1.3)
		dummyGraphs_per_class[key].SetMarkerStyle(markers_per_class[key])
		dummyGraphs_per_class[key].SetMarkerColor(colors_per_class[key])
		
		legend_aspectRatioVSSNR.AddEntry(dummyGraphs_per_class[key], key, "P")

	legend_aspectRatioVSSNR.Draw("same")

	# - Draw max beam size vs SNR
	logger.info("Plotting max beam size VS SNR ...")
	plot_maxSizeVSSNR= ROOT.TCanvas("plot_maxSizeVSSNR","plot_maxSizeVSSNR")
	plot_maxSizeVSSNR.cd()

	plotBkg_maxSizeVSSNR= ROOT.TH2D("plotBkg_maxSizeVSSNR","",100,-2,5,100,-1,2)
	plotBkg_maxSizeVSSNR.GetXaxis().SetTitle("log_{10}(SNR)")
	plotBkg_maxSizeVSSNR.GetYaxis().SetTitle("log_{10}(maxSize)")
	plotBkg_maxSizeVSSNR.Draw()
	
	snrVSmaxSizeGraph.SetMarkerSize(0.1)
	snrVSmaxSizeGraph.SetMarkerColor(ROOT.kWhite)
	snrVSmaxSizeGraph.GetXaxis().SetTitle("log_{10}(SNR)")
	snrVSmaxSizeGraph.GetYaxis().SetTitle("log_{10}(maxSize)")
	#snrVSmaxSizeGraph.Draw("AP")

	legend_maxSizeVSSNR= ROOT.TLegend(0.7,0.7,0.9,0.9)
	legend_maxSizeVSSNR.SetFillColor(0)
	legend_maxSizeVSSNR.SetTextFont(52)
	legend_maxSizeVSSNR.SetTextSize(0.04)

	for key in snrVSmaxSizeGraphs_per_class:
		snrVSmaxSizeGraphs_per_class[key].SetMarkerSize(marker_sizes_per_class[key])
		snrVSmaxSizeGraphs_per_class[key].SetMarkerStyle(markers_per_class[key])
		snrVSmaxSizeGraphs_per_class[key].SetMarkerColor(colors_per_class[key])
		snrVSmaxSizeGraphs_per_class[key].Draw("P")

		legend_maxSizeVSSNR.AddEntry(dummyGraphs_per_class[key], key, "P")

	legend_maxSizeVSSNR.Draw("same")



	# - Draw aspect ratio VS max beam size
	logger.info("Plotting aspect ratio vs max beam size ...")
	plot_maxSizeVSaspectRatio= ROOT.TCanvas("plot_maxSizeVSaspectRatio","plot_maxSizeVSaspectRatio")
	plot_maxSizeVSaspectRatio.cd()

	plotBkg_maxSizeVSaspectRatio= ROOT.TH2D("plotBkg_maxSizeVSaspectRatio","",100,-2,5,100,-1,2)
	plotBkg_maxSizeVSaspectRatio.GetXaxis().SetTitle("log_{10}(maxSize)")
	plotBkg_maxSizeVSaspectRatio.GetYaxis().SetTitle("log_{10}(maxSize/minSize)")
	plotBkg_maxSizeVSaspectRatio.Draw()
	
	maxSizeVSaspectRatioGraph.SetMarkerSize(0.1)
	maxSizeVSaspectRatioGraph.SetMarkerColor(ROOT.kWhite)
	maxSizeVSaspectRatioGraph.GetXaxis().SetTitle("log_{10}(SNR)")
	maxSizeVSaspectRatioGraph.GetYaxis().SetTitle("log_{10}(maxSize)")
	#maxSizeVSaspectRatioGraph.Draw("AP")

	legend_maxSizeVSaspectRatio= ROOT.TLegend(0.7,0.7,0.9,0.9)
	legend_maxSizeVSaspectRatio.SetFillColor(0)
	legend_maxSizeVSaspectRatio.SetTextFont(52)
	legend_maxSizeVSaspectRatio.SetTextSize(0.04)

	for key in maxSizeVSaspectRatioGraphs_per_class:
		maxSizeVSaspectRatioGraphs_per_class[key].SetMarkerSize(marker_sizes_per_class[key])
		maxSizeVSaspectRatioGraphs_per_class[key].SetMarkerStyle(markers_per_class[key])
		maxSizeVSaspectRatioGraphs_per_class[key].SetMarkerColor(colors_per_class[key])
		maxSizeVSaspectRatioGraphs_per_class[key].Draw("P")

		legend_maxSizeVSaspectRatio.AddEntry(dummyGraphs_per_class[key], key, "P")

	legend_maxSizeVSaspectRatio.Draw("same")




	# - Draw max size wrt img size VS unrotated bbox dim ratio
	logger.info("Plotting max size wrt img size VS unrotated bbox dim ratio ...")
	plot_maxImgSizeVSbboxDimRatio= ROOT.TCanvas("plot_maxImgSizeVSbboxDimRatio","plot_maxImgSizeVSbboxDimRatio")
	plot_maxImgSizeVSbboxDimRatio.cd()

	plotBkg_maxImgSizeVSbboxDimRatio= ROOT.TH2D("plotBkg_maxImgSizeVSbboxDimRatio","",100,-0.1,1.1,100,0.9,10)
	plotBkg_maxImgSizeVSbboxDimRatio.GetXaxis().SetTitle("maxBBoxSize/imgSize")
	plotBkg_maxImgSizeVSbboxDimRatio.GetYaxis().SetTitle("maxBBoxSize/minBBoxSize")
	plotBkg_maxImgSizeVSbboxDimRatio.Draw()
	
	maxImgSizeVSbboxDimRatioGraph.SetMarkerSize(0.1)
	maxImgSizeVSbboxDimRatioGraph.SetMarkerColor(ROOT.kWhite)
	maxImgSizeVSbboxDimRatioGraph.GetXaxis().SetTitle("maxBBoxSize/imgSize")
	maxImgSizeVSbboxDimRatioGraph.GetYaxis().SetTitle("maxBBoxSize/minBBoxSize")
	#maxImgSizeVSbboxDimRatioGraph.Draw("AP")

	legend_maxImgSizeVSbboxDimRatio= ROOT.TLegend(0.7,0.7,0.9,0.9)
	legend_maxImgSizeVSbboxDimRatio.SetFillColor(0)
	legend_maxImgSizeVSbboxDimRatio.SetTextFont(52)
	legend_maxImgSizeVSbboxDimRatio.SetTextSize(0.04)

	for key in maxImgSizeVSbboxDimRatioGraphs_per_class:
		maxImgSizeVSbboxDimRatioGraphs_per_class[key].SetMarkerSize(marker_sizes_per_class[key])
		maxImgSizeVSbboxDimRatioGraphs_per_class[key].SetMarkerStyle(markers_per_class[key])
		maxImgSizeVSbboxDimRatioGraphs_per_class[key].SetMarkerColor(colors_per_class[key])
		maxImgSizeVSbboxDimRatioGraphs_per_class[key].Draw("P")

		legend_maxImgSizeVSbboxDimRatio.AddEntry(dummyGraphs_per_class[key], key, "P")

	legend_maxImgSizeVSbboxDimRatio.Draw("same")



	# - Draw max size wrt img size VS unrotated bbox dim inverted ratio
	logger.info("Plotting max size wrt img size VS unrotated bbox dim inverted ratio ...")
	plot_maxImgSizeVSbboxDimInvRatio= ROOT.TCanvas("plot_maxImgSizeVSbboxDimInvRatio","plot_maxImgSizeVSbboxDimInvRatio")
	plot_maxImgSizeVSbboxDimInvRatio.cd()

	plotBkg_maxImgSizeVSbboxDimInvRatio= ROOT.TH2D("plotBkg_maxImgSizeVSbboxDimInvRatio","",100,-0.1,1.1,100,0.001,1.1)
	plotBkg_maxImgSizeVSbboxDimInvRatio.GetXaxis().SetTitle("maxBBoxSize/imgSize")
	plotBkg_maxImgSizeVSbboxDimInvRatio.GetYaxis().SetTitle("minBBoxSize/maxBBoxSize")
	plotBkg_maxImgSizeVSbboxDimInvRatio.Draw()
	
	maxImgSizeVSbboxDimInvRatioGraph.SetMarkerSize(0.1)
	maxImgSizeVSbboxDimInvRatioGraph.SetMarkerColor(ROOT.kWhite)
	maxImgSizeVSbboxDimInvRatioGraph.GetXaxis().SetTitle("maxBBoxSize/imgSize")
	maxImgSizeVSbboxDimInvRatioGraph.GetYaxis().SetTitle("minBBoxSize/maxBBoxSize")
	#maxImgSizeVSbboxDimInvRatioGraph.Draw("AP")

	legend_maxImgSizeVSbboxDimInvRatio= ROOT.TLegend(0.7,0.7,0.9,0.9)
	legend_maxImgSizeVSbboxDimInvRatio.SetFillColor(0)
	legend_maxImgSizeVSbboxDimInvRatio.SetTextFont(52)
	legend_maxImgSizeVSbboxDimInvRatio.SetTextSize(0.04)

	for key in maxImgSizeVSbboxDimInvRatioGraphs_per_class:
		maxImgSizeVSbboxDimInvRatioGraphs_per_class[key].SetMarkerSize(marker_sizes_per_class[key])
		maxImgSizeVSbboxDimInvRatioGraphs_per_class[key].SetMarkerStyle(markers_per_class[key])
		maxImgSizeVSbboxDimInvRatioGraphs_per_class[key].SetMarkerColor(colors_per_class[key])
		maxImgSizeVSbboxDimInvRatioGraphs_per_class[key].Draw("P")

		legend_maxImgSizeVSbboxDimInvRatio.AddEntry(dummyGraphs_per_class[key], key, "P")

	legend_maxImgSizeVSbboxDimInvRatio.Draw("same")





	# - Draw min size wrt to img size VS max size wrt img size 
	logger.info("Plotting min size wrt to img size VS max size wrt img size ...")
	plot_maxImgSizeVSminImgSize= ROOT.TCanvas("plot_maxImgSizeVSminImgSize","plot_maxImgSizeVSminImgSize")
	plot_maxImgSizeVSminImgSize.cd()

	plotBkg_maxImgSizeVSminImgSize= ROOT.TH2D("plotBkg_maxImgSizeVSminImgSize","",100,-0.1,1.1,100,-0.1,1.1)
	plotBkg_maxImgSizeVSminImgSize.GetXaxis().SetTitle("minBBoxSize/imgSize")
	plotBkg_maxImgSizeVSminImgSize.GetYaxis().SetTitle("maxBBoxSize/imgSize")
	plotBkg_maxImgSizeVSminImgSize.Draw()
	
	maxImgSizeVSminImgSizeGraph.SetMarkerSize(0.1)
	maxImgSizeVSminImgSizeGraph.SetMarkerColor(ROOT.kWhite)
	maxImgSizeVSminImgSizeGraph.GetXaxis().SetTitle("minBBoxSize/imgSize")
	maxImgSizeVSminImgSizeGraph.GetYaxis().SetTitle("maxBBoxSize/imgSize")
	#maxImgSizeVSminImgSizeGraph.Draw("AP")

	legend_maxImgSizeVSminImgSize= ROOT.TLegend(0.7,0.7,0.9,0.9)
	legend_maxImgSizeVSminImgSize.SetFillColor(0)
	legend_maxImgSizeVSminImgSize.SetTextFont(52)
	legend_maxImgSizeVSminImgSize.SetTextSize(0.04)

	for key in maxImgSizeVSminImgSizeGraphs_per_class:
		maxImgSizeVSminImgSizeGraphs_per_class[key].SetMarkerSize(marker_sizes_per_class[key])
		maxImgSizeVSminImgSizeGraphs_per_class[key].SetMarkerStyle(markers_per_class[key])
		maxImgSizeVSminImgSizeGraphs_per_class[key].SetMarkerColor(colors_per_class[key])
		maxImgSizeVSminImgSizeGraphs_per_class[key].Draw("P")

		legend_maxImgSizeVSminImgSize.AddEntry(dummyGraphs_per_class[key], key, "P")

	legend_maxImgSizeVSminImgSize.Draw("same")

	# - Draw bbox size plot
	plot_bboxWidthVSbboxHeight= ROOT.TCanvas("plot_bboxWidthVSbboxHeight","plot_bboxWidthVSbboxHeight")
	plot_bboxWidthVSbboxHeight.cd()

	plotBkg_bboxWidthVSbboxHeight= ROOT.TH2D("plotBkg_bboxWidthVSbboxHeight","",100,0.1,135,100,0.1,135)
	plotBkg_bboxWidthVSbboxHeight.GetXaxis().SetTitle("bbox_width (pix)")
	plotBkg_bboxWidthVSbboxHeight.GetYaxis().SetTitle("bbox_height (pix)")
	plotBkg_bboxWidthVSbboxHeight.Draw()

	legend_bboxWidthVSbboxHeight= ROOT.TLegend(0.7,0.7,0.9,0.9)
	legend_bboxWidthVSbboxHeight.SetFillColor(0)
	legend_bboxWidthVSbboxHeight.SetTextFont(52)
	legend_bboxWidthVSbboxHeight.SetTextSize(0.04)

	for key in bboxWidthVSbboxHeightGraphs_per_class:
		bboxWidthVSbboxHeightGraphs_per_class[key].SetMarkerSize(marker_sizes_per_class[key])
		bboxWidthVSbboxHeightGraphs_per_class[key].SetMarkerStyle(markers_per_class[key])
		bboxWidthVSbboxHeightGraphs_per_class[key].SetMarkerColor(colors_per_class[key])
		bboxWidthVSbboxHeightGraphs_per_class[key].Draw("P")

		legend_bboxWidthVSbboxHeight.AddEntry(dummyGraphs_per_class[key], key, "P")

	legend_bboxWidthVSbboxHeight.Draw("same")

	# - Wait until input exit is given (otherwise canvas will close)
	logger.info("Write 1 to exit...")
	x= input()
	if x==1:
		print ('Exiting ...')
		return 0

	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

