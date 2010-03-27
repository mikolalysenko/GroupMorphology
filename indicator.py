#################################################################
# Indicator operators and primitives
#################################################################
import scipy as sp;

#Performs a level set (thresholding) operation on the given function
def to_ind(f, alpha=0.):
	return sp.array(f > alpha, sp.dtype('float'));

#Union
def union(f,g):
	return to_ind(f + g);

#Intersection
def intersect(f,g):
	return to_ind(f * g);

#Complement
def complement(f):
	return to_ind(1. - f);

#Volume (aka L1 norm)
def vol(f):
	return sum(abs(f.flatten()));

