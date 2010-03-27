#################################################################
# SE(n) acting on Rn via group morphology
#################################################################

import scipy as sp;
import math;
import scipy.signal as signal;
import r_mor   as rn;
from indicator import *;


#################################################################
# Pointwise Group/Homogeneous space operators (untested)
#################################################################


#Group element product
def productp(g, h):
	c = math.cos(g[0]);
	s = math.sqrt(1. - c*c);
	return (g[0] + h[0], g[1] + c*h[1] - s*h[2], g[2] + s*h[1] + c*h[2]);

#Group element inverse
def inversep(g):
	c = math.cos(g[0]);
	s = math.sqrt(1. - c*c);
	return (-g[0], c*g[1] + s*g[2], -s*g[1] + c*g[2]);

#Applies the action g = (theta,x,y) to an element in R2
def actionp(g, p):
	c = math.cos(g[0]);
	s = math.sqrt(1. - c*c);
	return (g[1] + c*p[0] - s*p[1], g[2] + s*p[0] + c*p[1]);

#Projects an element g in SE(2) down to a point in R2
def projectp(g):
	return action(g, (0,0));

#Maps a point p in R2 to an element in SE(2)
def taup(p):
	return (0, p[0], p[1]);
	
#Gamma map
def gammap(g):
	return g[0];


#################################################################
# Index conversion (untested)
#################################################################

#Converts an indexed element (i,j,k) to an element of SE(2), given by (theta, x,y) according to the specified grid resolution dims
def index2group(index, dims):
	return (tr[2] * 2. * sp.pi / float(dims[2]), dims[0]/2 - index[0] , index[1]/2 - tr[1]);
	
#Converts a group element to an index
def group2index(g, dims):
	i,j = int(g[1] - dims[0]/2), int(g[2] - dims[1]/2);
	k = int(.5 * g[0] * dims[0] / sp.pi);
	if(k < 0):
		k += (int(k/dims[0]) + 1) * dims[0];
	return (i,j,k%dims[0]);


#################################################################
# Group-function primitives (untested)
#################################################################

#Computes the support of a function in SE(2)
def supp(f):
	return vol(f);

#Applies a rotation to the function f
def rotate(theta, f):
	return scipy.misc.imrotate(f, theta * 180. / pi);

#Group product on functions over the group (left action)
def product_l(g, f):
	res = sp.zeros((f.shape[0]+abs(g[1]), f.shape[1] + abs(g[2]), f.shape[3]));
	for r in range(f.shape[2]):
		theta = float(r) * pi / float(f.shape[2]);
		res[:,:,r] = r2.shift((g[1], g[2]), rotate(g[0], f));
	return res;

#Group product on functions over the group (right action)
def product_r(f, g):
	res = sp.zeros(f.shape);
	assert(False);

#Computes inversion of functions on the group
def inversion(f):
	res = sp.zeros(f.shape);
	for r in range(f.shape[2]):
		res[:,:,r] = f[:,:,(f.shape[2] - r)%f.shape[2]];
	return res;

#Applies the action g = (theta,x,y) to an indicator map f in R2
def action(g, f):
	return r2.shift((g[1], g[2]), rotate(g[0], f));

#Lifts the function f on R2 to a full function on SE(2) sampled at resolution R
def lift(f, R):
	res = sp.zeros((f.shape[0], f.shape[1], R));
	for r in range(R):
		theta = float(r) * 2. * pi / float(R);
		res[:,:,r] = rotate(theta, f);
	return res;

#Projects a function on SE(2) down to the homogeneous space about the given origin
def project(f):
	res = sp.zeros((f.shape[0], f.shape[1]));
	for x in range(f.shape[0]):
		for y in range(f.shape[1]):
			res[x,y] = sum(f[x,y,:].flatten());
	return res;

#The map tau from the paper
def tau(s, R):
	res = sp.zeros((s.shape[0], s.shape[1], R));
	res[:,:,0] = s;
	return s;
	
#The map gamma from the paper
def gamma(f):
	res = sp.zeros((f.shape[2]));
	for r in range(f.shape[2]):
		res[r] = sum(f[:,:,r].flatten());
	return res;


#################################################################
# Convolution operators
#################################################################

#Computes the full SE(2) convolution via the homogeneous method
# Not listed in paper due to space considerations, but derivation is easy
def conv(f, g):
	res = np.zeros((f.shape[0] + g.shape[0] - 1, \
					f.shape[1] + g.shape[1] - 1, \
					f.shape[2]));
	tmp = np.zeros((g.shape[0], g.shape[1]));	
	for r in range(f.shape[2]):
		theta_r = 360. * float(r)/float(f.shape[2]);
		tmp[:,:] = 0.;
		for q in range(f.shape[2]):
			theta_q = 360.*float(q)/float(g.shape[2]);
			tmp += sp.misc.imrotate(g[:,:,q], theta_q - theta_r);
		res[:,:,r] += r2.conv(f[:,:,r], tmp);
	del tmp;
	return res;

#Computes the convolution of a function on SE(2) acting on a function in R2
# (From cor 2)
def act_conv(f, s):
	res = np.zeros((s.shape[0]+f.shape[0]-1,s.shape[1]+f.shape[1]-1));
	for r in range(f.shape[2]):
		theta = r / float(f.shape[2]) * 360.;
		res += r2.conv(s, scipy.misc.imrotate(f[:,:,r], -theta));
	return res;

#Computes a homogeneous convolution over SE(2) between two functions in R2 at R uniform rotational samples
# (From cor 3)
def hg_conv(s, t, R):
	res = np.zeros((s.shape[0] + t.shape[0] - 1, s.shape[1] + t.shape[1] - 1, R));
	for r in range(R):
		theta = r / float(R) * 360.;
		res[:,:,r] = r2.conv(t, scipy.misc.imrotate(s, -theta));
	return r;




#################################################################
# Morphological operators
#################################################################

#Minkowski product
def mink_prod(f,g):
	return to_ind(conv(f,g));
	
#Left Minkowski quotient
def mink_quot_l(f,g):
	return to_ind(conv(f,g), supp(f) - 0.001);
	
#Right Minkowski quotient	
def mink_quot_r(f,g):
	return to_ind(conv(f,g), supp(g) - 0.001);
	
#Dilation
def dilation(f,s):
	return to_ind(act_conv(f,s));

#Erosion
def erosion(f,s):
	return to_ind(act_conv(f,s), supp(f) - 0.001);
	
#Homogeneous product
def hg_prod(s,t,R):
	return to_ind(hg_conv(s,t,R));
	
#Left Homogeneous quotient
def hg_quot_l(s,t,R):
	return to_ind(hg_conv(s,t,R), r2.supp(s) - 0.001);

#Right Homogeneous quotient
def hg_quot_r(s,t,R):
	return to_ind(hg_conv(s,t,R), r2.supp(t) - 0.001);


