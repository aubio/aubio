from aubio.aubioclass import *

def get_onset_mode(nvalue):
	""" utility function to convert a string to aubio_onsetdetection_type """
	if   nvalue == 'complexdomain' or nvalue == 'complex' :
		 return aubio_onset_complex
	elif nvalue == 'hfc'           :
		 return aubio_onset_hfc
	elif nvalue == 'phase'         :
		 return aubio_onset_phase
	elif nvalue == 'specdiff'      :
		 return aubio_onset_specdiff
	elif nvalue == 'energy'        :
		 return aubio_onset_energy
	elif nvalue == 'kl'            :
		 return aubio_onset_kl
	elif nvalue == 'mkl'           :
		 return aubio_onset_mkl
	elif nvalue == 'specflux'      :
		 return aubio_onset_specflux
	elif nvalue == 'dual'          :
		 return 'dual'
	else:
		 import sys
		 print "unknown onset detection function selected: %s" % nvalue
		 sys.exit(1)

def check_onset_mode(option, opt, value, parser):
	""" wrapper function to convert a list of modes to 
		aubio_onsetdetection_type """
	nvalues = parser.rargs[0].split(',')
	val =  []
	for nvalue in nvalues:
		val.append(get_onset_mode(nvalue))
		setattr(parser.values, option.dest, val)

def check_pitch_mode(option, opt, value, parser):
	""" utility function to convert a string to aubio_pitchdetection_type"""
	nvalues = parser.rargs[0].split(',')
	val = []
	for nvalue in nvalues:
		val.append(get_pitch_mode(nvalue))
		setattr(parser.values, option.dest, val)
