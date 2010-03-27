#################################################################
# Rn morphological operators
#
#  This is basically just a wrapper for various ndimage
#
#################################################################

#Miscellaneous functions for managing discrete Rn morphology
import scipy 		 as sp;
import scipy.ndimage as ndi;

from indicator import *;

#Computes the support of f
def supp(f):
	return vol(f);

#Computes the inverse of f
def invert(f):
	rs = [];
	for d in range(len(f.shape)):
		rs.append(slice(f.shape[d]-1, 0, -1));
	return f[tuple(rs)];
	
#Shifts the function f by the element v in R2
def shift(v, f):
	return ndi.shift(f, v);
	
#Convolution over Rn
def conv(f, g):
	ub = tuple(array(f.shape) + array(g.shape) - 1);
	return ndi.filters.convolve(pad(f,ub), pad(g,ub), mode='wrap');
	
#Correlation/conjugate convolution over Rn
def hg_conv(f, g):
	return conv(f, invert(g));

#Minkowski sum over Rn
def mink_add(f,g):
	return to_ind(r2_conv(f,g));
	
#Minkowski difference over Rn
def mink_sub(f,g):
	return to_ind(r2_conv(f,g), supp(g) - 0.001);

