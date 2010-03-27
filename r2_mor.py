#################################################################
# R2 Morphological operators
#################################################################

#Miscellaneous functions for managing discrete R2 morphology
import scipy as sp;
import scipy.signal as signal;
from indicator import *;

#Loads an image from file, converts it to an indicator function
def load_img(path):
	return to_ind(sp.misc.impread(path, True));

#Pads an indicator function to size nd, keeping it centered
def pad(f, nd):
	assert(nd[0] > f.shape[0]);
	assert(nd[1] > f.shape[1]);
	res = sp.zeros(nd);
	c = ((nd[0] - f.shape[0]) / 2, (nd[1] - f.shape[1]) / 2);
	res[c[0]:(c[0]+img.shape[0]),c[1]:(c[1]+img.shape[1])] = img;
	return res;

#Computes the inverse of an indicator function
def invert(f):
	return scipy.misc.imrotate(f, 180.);
	
#Shifts the function f by the element v in R2
def shift(v, f):
	assert(False);
	
#Convolution over R2
def conv(f, g):
	return signal.convolve2d(pad_ind(f, s2), pad_ind(g, s2));

#Minkowski sum over R2
def mink_add(f,g):
	return to_ind(r2_conv(f,g));
	
#Minkowski difference over R2
def mink_sub(f,g):
	return to_ind(r2_conv(f,g), vol_ind(g) - 0.001);

