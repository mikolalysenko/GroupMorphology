from math import *
from numpy import *
from numpy.fft import *
from scipy import *

'''
Cartesian to Polar coordinate conversion

Uses Bresenham's algorithm to convert the image f into a polar sampled image
'''
def rect2polar( f, R ):
	#Check bounds on R
	assert(R > 0)

	xs, ys = f.shape
	x0 = floor(xs / 2)
	y0 = floor(ys / 2)
	
	fp = []
	
	#Initialize 0,1 as special cases
	fp.append(array([ f[x0, y0] ], f.dtype))
	
	if R >= 1:
		fp.append(array(
			[ f[x0,y0+1], f[x0+1,y0+1], f[x0+1,y0], f[x0+1,y0-1], f[x0,y0-1], f[x0-1,y0-1], f[x0-1,y0], f[x0-1,y0+1] ]))
			
	#Perform Bresenham interpolation
	for r in range(2, R):
		#Allocate result
		res = zeros((8 * r + 4), f.dtype)

		#Handle axial directions
		res[0+0*r] = f[x0,   y0+r]
		res[1+2*r] = f[x0+r, y0]
		res[2+4*r] = f[x0,   y0-r]
		res[3+6*r] = f[x0-r, y0]

		#Set up scan conversion process
		x = 0
		y = r
		s = 1 - r
		t = 1

		while x <= y:
			#Handle x-crossing
			x = x + 1
	
			res[    t+0] = f[x0+x,y0+y]
			res[2*r-t+1] = f[x0+y,y0+x]
			res[2*r+t+1] = f[x0+y,y0-x]
			res[4*r-t+2] = f[x0+x,y0-y]
			res[4*r+t+2] = f[x0-x,y0-y]
			res[6*r-t+3] = f[x0-y,y0-x]
			res[6*r+t+3] = f[x0-y,y0+x]
			res[8*r-t+4] = f[x0-x,y0+y]
			t = t + 1
	
			#Update status flag
			if  s < 0:
				s = s + 2 * x + 1
			elif x <= y:
				#Also handle y-crossing
				y = y - 1
				s = s + 2 * (x - y) + 1
	
				res[    t+0] = f[x0+x,y0+y]
				res[2*r-t+1] = f[x0+y,y0+x]
				res[2*r+t+1] = f[x0+y,y0-x]
				res[4*r-t+2] = f[x0+x,y0-y]
				res[4*r+t+2] = f[x0-x,y0-y]
				res[6*r-t+3] = f[x0-y,y0-x]
				res[6*r+t+3] = f[x0-y,y0+x]
				res[8*r-t+4] = f[x0-x,y0+y]
				t = t + 1
		fp.append(res)
        
	return fp


'''
Polar to Rectilinear coordinate conversion

Converts striped polar image into a Cartesian image
'''
def polar2rect( fp, xs, ys ):
	#Get size
	R = len(fp)
	assert(xs >= 2*R and ys >= 2*R)
	
	x0 = floor(xs/2)
	y0 = floor(ys/2)

	#Allocate result
	f = zeros((xs, ys), fp[0].dtype)
	
	#print(f.dtype)
	#print(fp[0].dtype)
	
	#Handle special cases
	f[x0, y0] = fp[0][0]

	if R >= 1:
		tmp = fp[1];
		f[x0,y0+1]   = tmp[0];
		f[x0+1,y0+1] = tmp[1];
		f[x0+1,y0]   = tmp[2];
		f[x0+1,y0-1] = tmp[3];
		f[x0,y0-1]   = tmp[4];
		f[x0-1,y0-1] = tmp[5];
		f[x0-1,y0]   = tmp[6];
		f[x0-1,y0+1] = tmp[7];

	#Perform scan conversion via Bresenham's algorithm
	for r in range(2, R):
		#Read circle values
		res = fp[r]
		
		#Set axial values
		f[x0,y0+r] = res[0+0*r]
		f[x0+r,y0] = res[1+2*r]
		f[x0,y0-r] = res[2+4*r]
		f[x0-r,y0] = res[3+6*r]
		
		#Begin Bresenham interpolation
		x = 0
		y = r
		s = 1 - r
		t = 1

		while x <= y:
			#Handle x-crossing
			x = x + 1
			
			f[x0+x,y0+y] = res[    t+0]
			f[x0+y,y0+x] = res[2*r-t+1]
			f[x0+y,y0-x] = res[2*r+t+1]
			f[x0+x,y0-y] = res[4*r-t+2]
			f[x0-x,y0-y] = res[4*r+t+2]
			f[x0-y,y0-x] = res[6*r-t+3]
			f[x0-y,y0+x] = res[6*r+t+3]
			f[x0-x,y0+y] = res[8*r-t+4]
			t = t + 1
			
			if s < 0:
				s = s + 2 * x + 1
			elif x <= y:
				#Also handle y-crossing
				y = y - 1
				s = s + 2 * (x - y) + 1
				f[x0+x,y0+y] = res[    t+0]
				f[x0+y,y0+x] = res[2*r-t+1]
				f[x0+y,y0-x] = res[2*r+t+1]
				f[x0+x,y0-y] = res[4*r-t+2]
				f[x0-x,y0-y] = res[4*r+t+2]
				f[x0-y,y0-x] = res[6*r-t+3]
				f[x0-y,y0+x] = res[6*r+t+3]
				f[x0-x,y0+y] = res[8*r-t+4]
				t = t + 1
	return f


'''
Pads an array
'''
def __pad(f, ps):
	xs, ys = f.shape
	ff = zeros((2*ps+1, 2*ps+1), f.dtype)
	ff[:(xs/2)+(xs%2),:(ys/2)+(ys%2)] = f[(xs/2):, (ys/2):]
	ff[:(xs/2)+(xs%2),2*ps+1-(ys/2):] = f[(xs/2):, :(ys/2)]
	ff[2*ps+1-(xs/2):,:(ys/2)+(ys%2)] = f[:(xs/2), (ys/2):]
	ff[2*ps+1-(xs/2):,2*ps+1-(ys/2):] = f[:(xs/2), :(ys/2)]
	return ff


from scipy.misc.pilutil import *

def show_img(x):
	imshow(imresize(abs(x),[800,800]))
	return x

def show_vol(vol):
	for x in range(vol.shape[2]):
		show_img(vol[:,:,x])
	return vol


'''
Computes the SE(2) FFT of the function f
'''
def se2fft(f):
	#Get size bounds
	xs,ys,rs = f.shape
	ps = int(ceil(sqrt(xs*xs + ys*ys)))

	#Allocate f1
	fh = [];
	fh.append(zeros((rs,1), dtype('cfloat')))
	fh.append(zeros((rs,8), dtype('cfloat')))
	for p in range(2,ps):
		fh.append(zeros((rs,8*p+4), dtype('cfloat')))

	#Perform initial FFT pass
	for t in range(rs):
		for v in zip(range(ps), rect2polar(ifftshift(fft2(__pad(f[:,:,t],ps))),ps)):
			fh[v[0]][t,:] = v[1]

	return map(fft2, fh)


'''
Computes the inverse SE(2) FFT of fp onto an (xs,ys) grid
'''
def se2ifft(fp, xs, ys):
	ps = len(fp)
	rs = fp[0].shape[0]

	#Un-FFT input
	fp = map(ifft2, fp)

	#Perform polar-rect conversion
	f = zeros((xs, ys, rs), dtype('cfloat'))

	for t in range(rs):
		tmp = fftshift(ifft2(fftshift(polar2rect(map(lambda x : x[t,:], fp), 2*ps+1, 2*ps+1))))
		f[:,:,t] = tmp[(ps-floor(xs/2)):(ps+xs-ceil(xs/2)), (ps-floor(ys/2)):(ps+ys-ceil(ys/2))]
		
	return f



'''
Computes convolution of two functions using the brute force method
'''
def se2convolve_bf(f, g):
	assert(f.shape == g.shape);
	xs, ys, rs = f.shape;
	h = zeros((xs, ys, rs), dtype('cfloat'))
	for x0 in range(xs):
		for y0 in range(ys):
			for r0 in range(rs):
				cr = cos(2. * pi * float(r0) / float(rs))
				sr = sin(2. * pi * float(r0) / float(rs))
				for x1 in range(xs):
					for y1 in range(ys):
						tx =  cr * x1 + sr * y1
						ty = -sr * x1 + cr * y1
						tx = (int(tx) + x0) % xs
						ty = (int(ty) + y0) % ys
						for r1 in range(rs):
							h[x0,y0,r0] += f[x1, y1, r1] * g[tx, ty, (r1 - r0) % rs]
	return h

'''
Computes the convolution of f,g using the discrete Fourier method
'''
def se2convolve_df(f, g):
	assert(f.shape == g.shape);
	xs, ys, rs = f.shape;
	h = zeros((xs,ys,rs), dtype('cfloat'))
	return h;


def __gcd(a,b) :
	if b == 0 :
		return a
	return __gcd(b, a % b)

def __scalef(a, k, N):
	if k%N == 0:
		return cfloat(a)
	return (1. - exp(-2. * pi * sqrt(-1.) * float(k) / float(N))) / (1. - exp(-2. * pi * sqrt(-1.) * float(k) / float(N * a)))

'''
Interpolated matrix product (unoptimized)
'''
def interp_prod(f, g):
	#Get dimensions
	Nr, Nc = f.shape
	N = Nr * Nc / __gcd(Nr, Nc)
	
	a = N / Nr
	b = N / Nc
	
	print(Nr, Nc)
	
	def antialias(ff, i, j):
		return __scalef(a, i, N) * __scalef(b, j, N) * ff[(i+Nr) % Nr, (j+Nc) % Nc]
		
	
	#Allocate result
	h = zeros(f.shape, f.dtype)
	
	'''
	d_omega		= exp(-2. * pi * sqrt(-1.) / N)
	d_omega_n	= exp( 2. * pi * sqrt(-1.) / N)
	d_omega_a	= exp(-2. * pi * sqrt(-1.) / (N * a))
	d_omega_b	= exp(-2. * pi * sqrt(-1.) / (N * b))
	d_omega_bn	= exp( 2. * pi * sqrt(-1.) / (N * b))


	nx = 1.;
	dx = 1.;
	'''
	
	for x in range(Nr):
		'''
		ny = 1.;
		dy = 1.;
		'''
		for y in range(Nc):
			'''
			nk0 = 1.;
			dk0 = 1.;
			
			nk1 = nx;
			dk1 = dx;
			
			nk2 = ny;
			dk2 = dy;
			'''
			for k in range(N):
				t0 = 0
				t1 = 0
				
				for p in range(a):
					m = x + p * Nr
					t0 += __scalef(a, m, N) * __scalef(a, k+m, N)
					
				for q in range(b):
					l = y + q * Nc
					t1 +=  __scalef(b,l-k, N)
						
				s = t0 * t1 * __scalef(b, k, N) / (a * b)
				'''
				n = (1. - nx) * (1. - nk0) * (1. - nk1) * (1. - nk2)
				d = (1. - dx) * (1. - dk0) * (1. - dk1) * (1. - dk2)
				
				if (abs(d) < 1e-10):
					n = 1.
					d = 1.
				
				nk0 *= d_omega
				nk1 *= d_omega
				nk2 *= d_omega_n
				
				dk0 *= d_omega_b
				dk1 *= d_omega_a
				dk2 *= d_omega_bn
				'''
				#s = __scalef(b,k,N) * __scalef(a,x,N) * __scalef(a,k+x,N) * __scalef(b, y - k, N) / (a * b)
				#s = n / (d * a * b)
				#s = 1.
				h[x,y] += s *  f[x,k%Nc] * g[(k + x)%Nr, (y + N - k)%Nc]
			#ny *= d_omega
			#dy *= d_omega_b
		#nx *= d_omega
		#dx *= d_omega_a
				
				
						#h[x,y] += antialias(f, m, k) * antialias(g, k + m, l - k)
						#h[x,y] += antialias(f, m, k - m) * antialias(g, k, l - k + m) #correct
						#h[x,y] += antialias(f, m, k) * antialias(g, k - m, k + l - 2 * m)
 
	return h
	

'''
SE(2) representation product
'''
def se2prod(fh, gh):
	return map(lambda v : interp_prod(v[0], v[1]), zip(fh, gh))


'''
SE(2) convolution product
'''
def se2convolve(f, g):
	assert(f.shape == g.shape)
	return se2ifft(se2prod(se2fft(g), se2fft(f)), f.shape[0], f.shape[1])


'''
Testing code
'''

def test_sefft():
	f = zeros((32,32,32))
	f[15,15,:] = 1
	g = zeros((32,32,32))
	g[5:10,5:10,0] = 1

	fh = se2fft(f)
	gh = se2fft(g)
	show_img(fh[5])
	show_img(gh[5])
	
	return se2prod(gh, fh)



'''
t = zeros((11,11,12))
for v in range(12):
	x = 5. + 2. * cos(float(v) / 12. * 2. * pi)
	y = 5. + 2. * sin(float(v) / 12. * 2. * pi)
	print(x,y)
	t[int(y),int(x),v]=1

th = sefft.se2fft(t)
'''

