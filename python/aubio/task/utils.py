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
	elif nvalue == 'dual'          :
		 return 'dual'
	else:
		 import sys
		 print "unknown onset detection function selected"
		 sys.exit(1)

def get_pitch_mode(nvalue):
	""" utility function to convert a string to aubio_pitchdetection_type """
	if   nvalue == 'mcomb'  :
		 return aubio_pitch_mcomb
	elif nvalue == 'yin'    :
		 return aubio_pitch_yin
	elif nvalue == 'fcomb'  :
		 return aubio_pitch_fcomb
	elif nvalue == 'schmitt':
		 return aubio_pitch_schmitt
	else:
		 import sys
		 print "error: unknown pitch detection function selected"
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

def check_pitchm_mode(option, opt, value, parser):
	""" utility function to convert a string to aubio_pitchdetection_mode """
	nvalue = parser.rargs[0]
	if   nvalue == 'freq'  :
		 setattr(parser.values, option.dest, aubio_pitchm_freq)
	elif nvalue == 'midi'  :
		 setattr(parser.values, option.dest, aubio_pitchm_midi)
	elif nvalue == 'cent'  :
		 setattr(parser.values, option.dest, aubio_pitchm_cent)
	elif nvalue == 'bin'   :
		 setattr(parser.values, option.dest, aubio_pitchm_bin)
	else:
		 import sys
		 print "error: unknown pitch detection output selected"
		 sys.exit(1)


