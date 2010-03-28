#################################################################
# Indicator functions, set operations and other helper utilities
#################################################################
from scipy import *;

#Performs a level set (thresholding) operation on the given function
def to_ind(f, alpha=0.):
	return array(f > alpha, dtype('float'));

#Loads an image from file, converts it to an indicator function in R2
def load_img(path):
	return to_ind(misc.imread(path, flatten=True));

#Pads an indicator function up to size nd, keeping it centered
def pad(f, nd):
	c = (array(nd,dtype('float')) / 2. - array(f.shape,dtype('float')) / 2.).round();
	rs = [];
	for d in range(len(f.shape)):
		rs.append(slice(c[d], c[d] + f.shape[d]));
	res = zeros(nd);
	res[tuple(rs)] = f;
	return res;
	
#Computes upper bound on dimensions of shapes
def upper_bound(a, b):
	return tuple( maximum(array(a), array(b)) );

#Union
def union(f,g):
	ub = upper_bound(f.shape, g.shape);
	return to_ind(pad(f,ub) + pad(g, ub));

#Intersection
def intersect(f,g):
	ub = upper_bound(f.shape, g.shape);
	return to_ind(pad(f,ub) * pad(g,ub));

#Complement
def complement(f):
	return to_ind(1. - f);
	
#Set difference
def subtract(f, g):
	return intersect(f, complement(g));

#Volume (aka Haar measure)
# ONLY VALID FOR UNIFORM SAMPLINGS, FOR NONUNIFORM GRIDS PREMULTIPLY BY VOLUME FORM
def vol(f):
	return sum(abs(f.flatten()));

