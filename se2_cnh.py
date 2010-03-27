#Commutative 

import scipy;
import scipy.signal;
import scipy.ndimage;
import enthought.mayavi.mlab as mlab;



def config_product(f, g, R):
	r = np.zeros((f.shape[0], f.shape[1], R));
	for i in range(R):
		r[:,:,i] = scipy.signal.fftconvolve(scipy.misc.imrotate(f, i / float(R) * 360.) > 0, g > 0, 'same');
	return r;


def supp(f):
	return sum(sum(scipy.array(f>0, scipy.dtype('float'))));

def config_plot(f, g, cs, S=5000):
	Nx, Ny, Nr = cs.shape;
	count = 0;
	result = scipy.zeros(f.shape, scipy.dtype('float'));
	for x in range(Nx):
		for y in range(Ny):
			for r in range(Nr):
				if(cs[x,y,r] > 0):
					if(count%S == 0):
						result += apply_config(g, (x,y,r), cs.shape);
					count += 1;
	return result;
	
import heapq;

def find_path(cspace, start_config, end_config):

	predecessor = scipy.zeros(cspace.shape, scipy.dtype('int,int,int'));
	to_visit = [];
	
	def add_state(st, pred):
		v = [0,0,0];
		d = 0;
		for k in range(3):
			v[k] = (st[k] + cspace.shape[k]) % cspace.shape[k];
			d += min(abs(end_config[k] - v[k]), \
					 abs(v[k] - end_config[k]))
		if(cspace[v[0],v[1],v[2]] == 0.):
			return;
		pp = predecessor[v[0],v[1],v[2]];
		if(pp[0] != 0 or pp[1] != 0 or pp[2] != 0):
			return;
		d += sp.random.rand();
		heapq.heappush(to_visit, (d, (v[0], v[1], v[2])) );
		predecessor[v[0], v[1], v[2]] = pred;
		
	add_state(start_config, start_config);
	
	while (len(to_visit) > 0):
		next = (heapq.heappop(to_visit));
		next = next[1];
		if(next == end_config):
			break;
		for k in range(3):
			tmp = [next[0], next[1], next[2]];
			tmp[k] = tmp[k] + 1;
			add_state((tmp[0], tmp[1], tmp[2]), next);
			tmp[k] = tmp[k] - 2;
			add_state((tmp[0], tmp[1], tmp[2]), next);
		
	
	if(predecessor[end_config[0], end_config[1], end_config[2]] == (0,0,0)):
		print 'failed to find path';
		return [];
	
	#Generate list in reverse order
	st = end_config;
	path = [];
	while(not (st[0] == start_config[0] and st[1] == start_config[1] and st[2] == start_config[2])):
		path.append(st);
		st = predecessor[st[0], st[1], st[2]];

	return path;

def plot_path(dims, path):
    res = scipy.zeros(dims);
    for pt in path:
        res[pt[0],pt[1],pt[2]] = 1.;
    return res;
    
   
def test_conv():
	obstacles = load_set("config_obstacles.png");
	mlab.imshow(obstacles);
	piano = load_set("config_piano.png");
	mlab.imshow(piano);
	cspace =  config_product(piano, obstacles, 64);
	cspace = scipy.array(cspace >= (supp(piano)*.99), scipy.dtype('float'));
	volsf = mlab.pipeline.scalar_field(cspace);
	vol3d = mlab.pipeline.volume(volsf);
	img = config_plot(obstacles, piano, cspace);
	mlab.imshow(img);
	return obstacles, piano, cspace, img;

import time;
def test_conv2():
	obstacles = load_set("config_obstacles.png");
	piano = load_set("config_piano.png");
	
	t0 = time.clock();
	cspace =  config_product(piano, scipy.misc.imrotate(obstacles,180.), 64);
	t1 = time.clock();
	
	print (t1 - t0);
	
	print max(cspace.flatten());
	cspace = scipy.array(cspace >= max(cspace.flatten()) - 10, scipy.dtype('float'));
	return obstacles, piano, cspace;


def rescale(f, s):
	fs = scipy.array(scipy.misc.imresize(f, s) > 0., scipy.dtype('float'));
	if(s < 1.):
		res = scipy.zeros(f.shape, scipy.dtype('float'));
		x0 = (f.shape[0] - fs.shape[0]) / 2;
		x1 = x0 + fs.shape[0];
		y0 = (f.shape[1] - fs.shape[1]) / 2;
		y1 = y0 + fs.shape[1];
		res[x0:x1,y0:y1] = fs;
		print f.shape, res.shape;
		return res;
	else:
		x0 = (fs.shape[0] - f.shape[0]) / 2;
		x1 = x0 + f.shape[0];
		y0 = (fs.shape[1] - f.shape[1]) / 2;
		y1 = y0 + f.shape[1];
		return fs[x0:x1,y0:y1];

def scale_product(f, g, S, Smin, Smax):
	r = np.zeros((f.shape[0], f.shape[1], S));
	for i in range(S):
		t = float(i) / float(S) * (Smax - Smin) + Smin;
		s = sp.power(2., t);
		print s, t;
		fs = rescale(f, s);
		r[:,:,i] = scipy.signal.fftconvolve(fs, g, 'same');
		print max(r[:,:,i].flatten());
	return r;


def test_simil():
	shape = load_set("shape2.png");
	simil =  scale_product( \
		scipy.array(scipy.misc.imrotate(shape,180.)>0., scipy.dtype('float')), \
		shape, 64, -3, 0);	
	sf = mlab.pipeline.scalar_field(simil);
	#vol = mlab.pipeline.volume(sf);
	return shape, simil;



import scipy as sp;
def find_pts(cspace):
    res = [];
    for x in range(100):
        h = [sp.random.randint(512), sp.random.randint(512), sp.random.randint(64)];
        if(cspace[h[0],h[1],h[2]] > 0.):
            res.append((h[0], h[1], h[2]));
    return res; 
            
def test_path(cspace):
    v = find_pts(cspace);
    print v;
    return find_path(cspace, v[0], v[1]);
    
def test_path2():
	obst,piano,cspace = test_conv2();
	path = test_path(cspace);
	return obst,piano,cspace,path;


def test3():
	#Start = cspace[128+220,128+150,63]
	#End = cspace[128+30,128+230,32];
	obst,piano,cspace = test_conv2();
	path = find_path(cspace,(128+256-30,128+256-230,32),(128+256-220,128+256-150,63));
	return obst,piano,cspace,path;


def render_path(obst,piano,cspace,path):
	result = scipy.zeros(obst.shape, scipy.dtype('float'));
	for c in path:
		result += apply_config(piano, c, cspace.shape);
	return result;



