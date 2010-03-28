#################################################################
# Rn morphological operators
#
#  This is basically just a wrapper for various ndimage
#
#################################################################

#Miscellaneous functions for managing discrete Rn morphology
import scipy 		 	as sp;
import scipy.ndimage 	as ndi;

from numpy.fft import rfftn, irfftn, fftshift, ifftshift;
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
	
#Pads an indicator function up to size nd, recentering to origin
def cpad(f, nd):
	return fftshift(pad(f,nd));
	
#Convolution over Rn
def conv(f, g):
	ub = tuple(array(f.shape) + array(g.shape) - 1);
	fh = rfftn(cpad(f,ub));
	gh = rfftn(cpad(g,ub));
	res = ifftshift(irfftn(fh * gh));
	del fh, gh;
	return res;
	
#Correlation/homogeneous convolution over Rn
def hg_conv(f, g):
	ub = tuple(array(f.shape) + array(g.shape) - 1);
	fh = rfftn(cpad(f,ub));
	gh = rfftn(cpad(g,ub));
	res = ifftshift(irfftn(fh * sp.conjugate(gh)));
	del fh, gh;
	return res;

#Minkowski sum over Rn
def mink_sum(f,g):
	return to_ind(conv(f,g), 1);
	
#Minkowski difference over Rn
def mink_diff(f,g):
	return to_ind(conv(f,g), supp(g) - 1);
	
#Homogeneous Minkowski sum
def hg_mink_sum(f,g):
	return to_ind(hg_conv(f, g), 1);

#Homogeneous Minkowski left difference
def hg_mink_ldiff(f,g):
	return to_ind(conv(f,g), supp(g) - 1);
	
#Homogeneous Minkowski right difference
def hg_mink_rdiff(f,g):
	return to_ind(conv(f,g), supp(f) - 1);


