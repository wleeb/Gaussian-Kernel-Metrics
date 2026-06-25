import finufft as fi
import time as tm
import wprin as wp
import numpy as np
import numpy.fft as dft
import matplotlib.pyplot as plt
import time
import numpy.random as rnd

#
#
#

def main():
    wp.prini(13,1)

    test_gk2d()

    exit()

#
#
#

def test_gk2d():

    xs = np.empty(200000,dtype='float64')
    ys = np.empty(200000,dtype='float64')


    ts = np.empty(10000,dtype='float64')
    fs = np.empty([5000,5000],dtype='float64')
    gs = np.empty([5000,5000],dtype='float64')

    fconvs = np.empty([5000,5000],dtype='float64')

    fzs = np.empty([5000,5000],dtype='complex128')
    gzs = np.empty([5000,5000],dtype='complex128')
    fzs0 = np.empty(1000,dtype='complex128')
    fzs1 = np.empty(1000,dtype='complex128')


    freqs = np.empty(10000,dtype='float64')
    hs = np.empty([1000,1000],dtype='float64')
    ws = np.empty(10000,dtype='float64')


    fff = np.empty(100000,dtype='float64')
    ggg = np.empty(100000,dtype='float64')


    fpols2 = np.empty([10000,1000],dtype='float64')
    gpols2 = np.empty([10000,1000],dtype='float64')


    fpols = np.empty([10000,1000],dtype='float64')
    gpols = np.empty([10000,1000],dtype='float64')
    hpols = np.empty([10000,1000],dtype='float64')


    fzs_pol = np.empty([15000,5000],dtype='complex128')
    gzs_pol = np.empty([15000,5000],dtype='complex128')


    angls = np.empty(100000,dtype='float64')
    rs = np.empty(100000,dtype='float64')
    rfs = np.empty(100000,dtype='float64')

    cen = np.empty(10,dtype='float64')
    cen2 = np.empty(10,dtype='float64')

    cens = np.empty([1000,2],dtype='float64')
    cens2 = np.empty([1000,2],dtype='float64')
    heis = np.empty(1000,dtype='float64')
    sigs = np.empty(1000,dtype='float64')

    cens3d = np.empty([1000,3],dtype='float64')
    cens3d_rot = np.empty([1000,3],dtype='float64')


    heis2 = np.empty(1000,dtype='float64')
    sigs2 = np.empty(1000,dtype='float64')


    prods = np.empty(20000,dtype='float64')
    prods2 = np.empty(20000,dtype='float64')

    prods_pad = np.empty(20000,dtype='float64')


    prods_all = np.empty([100,10000],dtype='float64')

    dists = np.empty(20000,dtype='float64')
    dists2 = np.empty(20000,dtype='float64')


    gkdists = np.empty(50000,dtype='float64')
    gkdists2 = np.empty(50000,dtype='float64')
    gkdists3 = np.empty(50000,dtype='float64')


    rotx = np.empty([3,3],dtype='float64')
    roty = np.empty([3,3],dtype='float64')
    rotz = np.empty([3,3],dtype='float64')
    rotz2 = np.empty([3,3],dtype='float64')

    rmat = np.empty([3,3],dtype='float64')





    time0 = tm.time()

    r=2.684
    r=6.184
###    r=10.684
###    r=2.584
    wp.prin2_long('r=',r,1)


    ngaus=5
    generate_pars_dumb(cens,heis,sigs,ngaus)

    wp.prinr2('cens=',cens,ngaus,2)


    deps=1e-16


    n = nmax_mixture(r,sigs,ngaus,deps)

###    n=n+1
###    n=2*n

    n=2*np.int(n/2 + 10)

    n=n+200

    n=200
###    n=2*n

    wp.prinf('n=',n,1)
###    exit()

    na = n
    na=2*na
    na=2*na
    na=2*na


    nr = n
    nr = 2*nr
    nr = 2*nr
    nr = 2*nr

    wp.prinf('na=',na,1)
    wp.prinf('nr=',nr,1)


###    tau0 = (r/6)**2
    tau0 = .1


    alf = .5562
    p=1.651

    omg = 0.5

    wp.prin2('r=',r,1)
    wp.prin2('tau0=',tau0,1)
    wp.prin2('alf=',alf,1)
    wp.prin2('p=',p,1)
    wp.prin2('omg=',omg,1)

    nlevs=4
    wp.prinf('nlevs=',nlevs,1)




    for i in range(0,n+1):
        ts[i] = -r + 2*r*i/n
###        fs[i] = evalgaus333(ts[i],cen,wid)

    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)


    dt = 2*r / n
    wp.prin2('dt=',dt,1)

    dpi = np.pi
    wp.prin2('dpi=',dpi,1)


#
#    radial grid
#
###    na=np.int(1.2*n)
###    nr=2*n


#
#    rotate gaussians, for second mixture
#
    krot = np.int(n/2) + 8

    krot = 4

###    theta = (krot * 2+1) * np.pi / na
###    theta = krot * 2* np.pi / na

    theta = 2*np.pi * .553021
    rotvecs_dumb(cens,ngaus,theta,cens2)


    wp.prinr2('cens2=',cens2,ngaus,2)

    ii=0
    hh = cens[ii,0]**2 + cens[ii,1]**2
    hh2 = cens2[ii,0]**2 + cens2[ii,1]**2

    wp.prin2('hh=',hh,1)
    wp.prin2('hh2=',hh2,1)


    sigs2[0:ngaus] = sigs[0:ngaus]
    heis2[0:ngaus] = heis[0:ngaus]

###    heis2[0] = heis[ngaus-1] + .001
###    heis2[ngaus-1] = heis[1] + .002
###    sigs2[0] = sigs[0] / 1.01
###    sigs2[1] = sigs[1] / 1.02


    ngaus2 = ngaus


    wp.prin2('sigs=',sigs,ngaus)
    wp.prin2('sigs2=',sigs2,ngaus)
    wp.prin2('heis=',heis,ngaus)
    wp.prin2('heis2=',heis2,ngaus)

    wp.prinf('ngaus=',ngaus,1)
    wp.prinf('ngaus2=',ngaus2,1)

###    exit()

###    deps = 1e-15
###    rf = rfmax_mixture(sigs,ngaus,deps)

###    wp.prin2('rf=',rf,1)


    rf = n / (4*r)

    wp.prin2('rf=',rf,1)


#
#    evaluate gaussian mixtures on cartesian grid
#
    evalmix2d_fast(ts,n,cens,sigs,heis,ngaus,fs)
    evalmix2d_fast(ts,n,cens2,sigs2,heis2,ngaus2,gs)

    wp.prin2('fs=',fs,n)
    wp.prin2('fs=',fs[:,0],n)

    wp.prin2('gs=',gs,n)
    wp.prin2('gs=',gs[:,0],n)


    na2 = 10*na


    rf = n / (4*r)
    dist_min = eucl_rots(fs,gs,n,nr,na,na2,r,rf,dists)
    wp.prin2('dist_min=',dist_min,1)


    gkmin = gk2d_rots(fs,gs,n,nr,na,na2,r,rf,p,alf,tau0,omg,nlevs,gkdists)

    deps_pad = 1e-12
    gkmin2 = gk2d_rots_pad(fs,gs,n,nr,na,na2,r,rf,p,alf,tau0,omg, \
        nlevs,deps_pad,gkdists)


    wp.prin2('gkmin=',gkmin,1)
    wp.prin2('gkmin2=',gkmin2,1)



    dist = gk2d_pad_fast(fs,gs,n,r,p,alf,tau0,omg,nlevs,deps_pad)
    dist2 = gk2d_nopad_fast(fs,gs,n,r,p,alf,tau0,omg,nlevs)


    wp.prin2('dist=',dist,1)
    wp.prin2('dist2=',dist2,1)

    return()

#
#
#

def evalmix2d_fast(ts,n,cens,wids,heis,ngaus,fs):
#
#                            description:
#
#    Evaluates samples of 2D gaussian mixture on n-by-n grid.
#    If centers are 3D, it will evaluate the tomographic projection
#    of the 3D mixture onto the x-y plane.
#
#
#                        input parameters:
#
#  ts - length n real array containing the grid points; mixture will be
#    evaluated on points (ts[i],ts[j]), 0 \le i,j \le n-1
#  n - the number of samples on each dimension
#  cens - real array of size ngaus-by-2 or ngaus-by-3
#  wids - length ngaus array containing scale parameters of each gaussian
#  heis - length ngaus array containing heights (weights) of each gaussian
#  ngaus - number of gaussians in the mixture
#
#
#                        output parameters:
#
#  fs - n-by-n array containing the gaussian mixture values
#

    tvec = np.empty(2,dtype='float64')
    ffs0 = np.empty([n,ngaus],dtype='float64')
    ffs1 = np.empty([n,ngaus],dtype='float64')
#
    difs0 = np.empty([n,ngaus],dtype='float64')
    difs1 = np.empty([n,ngaus],dtype='float64')
    gg = np.empty([n,n,ngaus],dtype='float64')


    dpi = np.pi

###    fs[0:n,0:n] = 0

#
#    evaluate 1D gaussians
#
    difs0[0:n,0:ngaus] = np.add.outer(ts[0:n] ,-cens[0:ngaus,0])
    difs1[0:n,0:ngaus] = np.add.outer(ts[0:n] ,-cens[0:ngaus,1])

###    wp.prinr2('difs0=',difs0,n,ngaus)
###    wp.prinr2('difs1=',difs1,n,ngaus)

###    ffs0[0:n,0:ngaus] = np.exp(-np.divide(difs0[0:n,0:ngaus]**2, \
###        wids[None,0:ngaus]))
    ffs0[0:n,0:ngaus] = np.exp(-np.divide(difs0[0:n,0:ngaus]**2, \
        wids[np.newaxis,0:ngaus]))
    ffs0[0:n,0:ngaus] = np.divide(ffs0[0:n,0:ngaus], \
        np.sqrt(dpi*wids[np.newaxis,0:ngaus]))
#
    ffs1[0:n,0:ngaus] = np.exp(-np.divide(difs1[0:n,0:ngaus]**2, \
        wids[np.newaxis,0:ngaus]))
    ffs1[0:n,0:ngaus] = np.divide(ffs1[0:n,0:ngaus], \
        np.sqrt(dpi*wids[np.newaxis,0:ngaus]))


#
#    outer products to get 2D gaussians, and sum
#
    gg[0:n,0:n,0:ngaus] = np.multiply(ffs0[0:n,np.newaxis,0:ngaus], \
        ffs1[np.newaxis,0:n,0:ngaus])

    fs[0:n,0:n] = np.sum(np.multiply(gg[0:n,0:n,0:ngaus], \
        heis[np.newaxis,np.newaxis,0:ngaus]),axis=2)

###    gg[0:n,0:n,0:ngaus] = np.einsum('ik,jk->ijk', \
###                   ffs0[0:n, 0:ngaus], \
###                   ffs1[0:n, 0:ngaus])
###    fs[0:n,0:n] = np.sum(gg[0:n, 0:n, 0:ngaus] * heis[0:ngaus],axis=2)


###    wp.prin2('gg=',gg[0:n,0:n,0:ngaus],n*n*ngaus)


###    wp.prin2('fs=',fs[:,0:n],n*n)


###    wp.prinr2('cens=',cens,ngaus,2)
###    wp.prin2('heis=',heis,ngaus)
###    wp.prin2('wids=',wids,ngaus)


    return()

#
#
#

def generate_pars_dumb(cens,heis,sigs,ngaus):
#
#    centers
#
    ii=0
    cens[ii,0] = -.11040
    cens[ii,1] = .1549

    cens[ii,0] = .0001
    cens[ii,1] = -.0002

    ii=ii+1
    cens[ii,0] = .04
    cens[ii,1] = -.14103

    ii=ii+1
    cens[ii,0] = .391
    cens[ii,1] = -.3591

    ii=ii+1
    cens[ii,0] = .299
    cens[ii,1] = -0.5101

    ii=ii+1
    cens[ii,0] = -0.25
    cens[ii,1] = .291

#
#    scales
#
    sigs[0] = .05
    sigs[1] = .045
    sigs[2] = .054
    sigs[3] = .093
    sigs[4] = .046

    sigs[0:ngaus] = sigs[0:ngaus]/2.5
###    sigs[0:ngaus] = 10*sigs[0:ngaus]

#
#    heights
#

#
    heis[0] = 0.55
    heis[1] = 0.91
    heis[2] = 0.951
    heis[3] = 0.8011
    heis[4] = 1.0555



    return()

#
#
#

def nmax_mixture(r,sigs,ngaus,deps):
#
#    evaluates the number of samples per dimension needed
#    to resolve gaussian mixture (to a specified precision);
#    returns an even value always
#
    dpi = np.pi


    sig_min = sigs[0]
    for i in range(0,ngaus):
        if (sigs[i] < sig_min):
            sig_min = sigs[i]

    n = 2 * r * np.sqrt(np.log(1/deps) / sig_min) / dpi

    n = np.int(n) + 1

#
#    double to satisfy Nyquist
#
    n = 2*n

###    n=3*n

###    wp.prinf('n=',n,1)

    wp.prin2('sigs=',sigs,ngaus)
    wp.prin2('sig_min=',sig_min,1)


    return(n)

#
#
#

def rfmax_mixture(sigs,ngaus,deps):
#
#    evaluates maximum frequency (up to the specified
#    precision) for a gaussian mixture
#
    dpi = np.pi

    sig_min = sigs[0]
    for i in range(0,ngaus):
        if (sigs[i] < sig_min):
            sig_min = sigs[i]


    wp.prin2('sigs=',sigs,ngaus)
    wp.prin2('sig_min=',sig_min,1)

    rf = np.sqrt(np.log(1/deps) / sig_min) / dpi

    return(rf)

#
#
#

def prodpolar_z(fpols,gpols,r,nr,na):

    rs = np.empty(nr+1,dtype='float64')

    rs[0:nr+1] = np.linspace(0,r,nr+1)

    wp.prin2('rs=',rs,nr+1)
    wp.prin2('r=',r,1)

    wp.prin2('r=',r,1)
    wp.prinf('nr=',nr,1)
    wp.prinf('na=',na,1)


    dr = r / nr
    da = 2*np.pi / na


    prod = np.sum(fpols[0:nr,0:na] * np.conj(gpols[0:nr,0:na]) \
        * rs[0:nr,np.newaxis]) * dr * da


    return(prod)

#
#
#

def evalmixn2d_fourier_polar(rs,angls,nr,na,cens,sigs,heis,ngaus,fzs_pol):
#
#    evaluates the fourier transform of a gaussian mixture
#    on a polar grid
#
    npts = nr*na

    xs = np.empty(npts,dtype='float64')
    ys = np.empty(npts,dtype='float64')

    pp_cos = np.empty([nr,na],dtype='float64')
    pp_sin = np.empty([nr,na],dtype='float64')

    fzs_list = np.empty(npts,dtype='complex128')

#
#    polar grid
#
    pp_cos[0:nr,0:na] = np.outer(rs[0:nr],np.cos(angls[0:na]))
    pp_sin[0:nr,0:na] = np.outer(rs[0:nr],np.sin(angls[0:na]))

    xs[0:npts] = np.reshape(pp_cos[0:nr,0:na],npts)
    ys[0:npts] = np.reshape(pp_sin[0:nr,0:na],npts)

#
#    evaluate mixture on polar grid
#
    evalmixn2d_fourier_list(xs,ys,npts,cens,sigs,heis,ngaus,fzs_list)
    fzs_pol[0:nr,0:na] = np.reshape(fzs_list[0:npts],[nr,na])




    return()

#
#
#

def evalmixn2d_fourier_list(xs,ys,n,cens,sigs,heis,ngaus,fzs):

    ffs0 = np.empty([n,ngaus],dtype='complex128')
    ffs1 = np.empty([n,ngaus],dtype='complex128')
#
    prods_x = np.empty([n,ngaus],dtype='float64')
    prods_y = np.empty([n,ngaus],dtype='float64')
#
    zzxs = np.empty([n,ngaus],dtype='complex128')
    zzys = np.empty([n,ngaus],dtype='complex128')



    dpi = np.pi
    zi = 1j

    ffs0[0:n,0:ngaus] = np.outer(xs[0:n]**2,dpi**2 * sigs[0:ngaus])
    ffs0[0:n,0:ngaus] = np.exp(-ffs0[0:n,0:ngaus])

    prods_x[0:n,0:ngaus] = np.outer(xs[0:n],cens[0:ngaus,0])
    zzxs[0:n,0:ngaus] = (np.cos(2*dpi*prods_x[0:n,0:ngaus]) \
        - zi*np.sin(2*dpi*prods_x[0:n,0:ngaus]))


    ffs1[0:n,0:ngaus] = np.outer(ys[0:n]**2,dpi**2 * sigs[0:ngaus])
    ffs1[0:n,0:ngaus] = np.exp(-ffs1[0:n,0:ngaus])

    prods_y[0:n,0:ngaus] = np.outer(ys[0:n],cens[0:ngaus,1])
    zzys[0:n,0:ngaus] = (np.cos(2*dpi*prods_y[0:n,0:ngaus]) \
        - zi*np.sin(2*dpi*prods_y[0:n,0:ngaus]))

    ffs0[0:n,0:ngaus] = ffs0[0:n,0:ngaus] * zzxs[0:n,0:ngaus]
    ffs1[0:n,0:ngaus] = ffs1[0:n,0:ngaus] * zzys[0:n,0:ngaus]


    fzs[0:n] = np.sum(ffs0[0:n,0:ngaus]*ffs1[0:n,0:ngaus]*heis[np.newaxis,0:ngaus], \
        axis=1)


    return()

#
#
#

def evalmix2d_fourier_dumb(sf,tf,cens,sigs,heis,ngaus):

    ffz = 0.0
    for i in range(0,ngaus):
        fincr = evalgausn2d_fourier(sf,tf,cens[i],sigs[i])
        ffz = ffz + fincr*heis[i]


    return(ffz)

#
#
#

def prodspin_nufft_sub(fs,gs,n,nr,na,na2,r,prods_pad):
#
#                            description:
#
#    Evaluates inner products between f and rotations of g from samples
#    of f and g on an n-by-n cartesian grid. The samples are interpolated
#    to a nr-by-na polar grid using the NUFFT. Inner products are computed
#    at na2 equispaced angles.
#
#
#                        input parameters:
#
#  fs,gs - n-by-n real arrays with samples of f, g on [-r,r)x[-r,r)
#  n - number of samples per dimensions; assumed to be an even integer (for now)
#  nr - number of samples in radial direction
#  na - number of samples in angular direction
#  na2 - number of inner products
#  r - length of each side
#
#
#                        output parameters:
#
#  prods_pad - na2-length real array containing the inner products between f
#    and each rotation of g
#

    xs = np.empty(na*nr,dtype='float64')
    ys = np.empty(na*nr,dtype='float64')
#
    pp_cos = np.empty([nr,na],dtype='float64')
    pp_sin = np.empty([nr,na],dtype='float64')
#
    gps_unwind_z = np.empty(nr*na,dtype='complex128')
#
    fps_unwind = np.empty(nr*na,dtype='float64')
    gps_unwind = np.empty(nr*na,dtype='float64')

#
    fps = np.empty([nr,na],dtype='float64')
    gps = np.empty([nr,na],dtype='float64')
#
    gzs = np.empty([n,n],dtype='complex128')
    gzs2 = np.empty([n,n],dtype='complex128')
#
    angls = np.empty(na+1,dtype='float64')
    rs = np.empty(nr+1,dtype='float64')



    deps = 1e-14

    wp.prin2('deps=',deps,1)

    wp.prin2('fs, inside=',fs,n)
    wp.prin2('gs, inside=',gs,n)

    wp.prinf('n=',n,1)
    wp.prinf('nr=',nr,1)
    wp.prinf('na=',na,1)


###    for i in range(0,nr+1):
###        rs[i] = i*r / nr

    rs[0:nr+1] = np.linspace(0,r,nr+1)

    wp.prin2('rs=',rs,nr+1)
    wp.prin2('r=',r,1)


    dpi = np.pi

###    for i in range(0,na+1):
###        angls[i] = 2*dpi *i/na

    angls[0:na+1] = np.linspace(0,2*dpi,na+1)

    wp.prin2('angls=',angls,na+1)
    wp.prin2('2pi=',2*dpi,1)


#
#    polar grid
#
    npts = nr*na

    pp_cos[0:nr,0:na] = np.outer(rs[0:nr],np.cos(angls[0:na]))
    pp_sin[0:nr,0:na] = np.outer(rs[0:nr],np.sin(angls[0:na]))

    nh = np.int(n/2)

#
#    interpolate f and g to nr-by-na polar grid
#
    xs[0:npts] = np.reshape(pp_cos[0:nr,0:na],npts) * np.pi / r
    ys[0:npts] = np.reshape(pp_sin[0:nr,0:na],npts) * np.pi / r


    interp_nufft(fs,n,r,xs,ys,npts,fps_unwind,deps)
    fps[0:nr,0:na] = np.reshape(fps_unwind[0:npts],[nr,na])

    interp_nufft(gs,n,r,xs,ys,npts,gps_unwind,deps)
    gps[0:nr,0:na] = np.reshape(gps_unwind[0:npts],[nr,na])


    wp.prin2('gps=',gps,10)
    wp.prin2('fps=',fps,10)


#
#    evaluate all inner products
#
    err_z = prodspin_polar(fps,gps,nr,na,na2,r,prods_pad)



    return()

#
#
#

def interp_nufft(fs,n,r,xs,ys,npts,fps_unwind,deps):
#
#    interpolates a function, with values on an n-by-n
#    cartesian grid, to arbitrary points (xs[i],ys[i])
#

    freqs = np.empty(n,dtype='float64')
#
    fzs = np.empty([n,n],dtype='complex128')
    fzs2 = np.empty([n,n],dtype='complex128')
#
    fps_unwind_z = np.empty(npts,dtype='complex128')


    df = 1/(2*r)

    nh = np.int(n/2)
    ftrs_2d_fft(fs,n,r,fzs,freqs)

    fzs2[0:nh,0:nh] = fzs[nh:n,nh:n]
    fzs2[0:nh,nh:n] = fzs[nh:n,0:nh]

    fzs2[nh:n,0:nh] = fzs[0:nh,nh:n]
    fzs2[nh:n,nh:n] = fzs[0:nh,0:nh]


    wp.prin2('freqs=',freqs,n)

    fps_unwind_z[0:npts] = fi.nufft2d2(xs[0:npts], ys[0:npts], \
        fzs2[0:n,0:n], eps=deps, isign=1)
    fps_unwind[0:npts] = np.real(fps_unwind_z[0:npts])*df**2
#


    if (0):

        wp.prinz('fzs=',fzs,n*n)
        wp.prinz('fzs2=',fzs2,n*n)
    
        hhh = np.sum(np.abs(fzs[0:n,0:n]))
        hhh2 = np.sum(np.abs(fzs2[0:n,0:n]))
    
        wp.prin2('hhh=',hhh,1)
        wp.prin2('hhh2=',hhh2,1)
        chk0 = hhh-hhh2
        wp.prin2('chk0=',chk0,1)

        for i in range(0,n):
            wp.prinz('fzs=',fzs[i,:],n)
        for i in range(0,n):
            wp.prinz('fzs2=',fzs2[i,:],n)
    
        exit()

    return()

#
#
#

def fsums_fast(fcoefs,m,n,ell,fs):
#
#                            description:
#
#    This code evaluates f from its Fourier series.
#    The code uses the IFFT and is vectorized.
#
#
#                        input parameters:
#
#  fcoefs - n length complex array, with Fourier coefficients in DFT ordering:
#
#          \what{f}[k] = \int_{0}^{L} f(t) \phi_k(t) dt             (1)
#
#    with k ranging from -n/2 to n/2-1
#  n - the number of samples; assumed to be an even integer (for now)
#  ell - interval length
#
#
#                        output parameters:
#
#  fs - n-length complex vector containing the n equispaced samples of f
#    on [0,L) (only the left endpoint f(a) is included, not f(b))
#

    fs[0:m,0:n] = dft.ifft(fcoefs[0:m,0:n],axis=1) * n / ell
###    fs[0:n] = np.real(fff[0:n])

    return()

    for i in range(0,n):
        fs[i] = 0
        for j in range(0,n):
            fs[i] = fs[i] + fcoefs[j]*np.exp(2*np.pi*1j*j*i/n) / ell

    return()

#
#
#

def fsum_fast(fcoefs,n,ell,fs):
#
#                            description:
#
#    This code evaluates f from its Fourier series.
#    The code uses the IFFT and is vectorized.
#
#
#                        input parameters:
#
#  fcoefs - n length complex array, with Fourier coefficients in DFT ordering:
#
#          \what{f}[k] = \int_{0}^{L} f(t) \phi_k(t) dt             (1)
#
#    with k ranging from -n/2 to n/2-1
#  n - the number of samples; assumed to be an even integer (for now)
#  ell - interval length
#
#
#                        output parameters:
#
#  fs - n-length complex vector containing the n equispaced samples of f
#    on [0,L) (only the left endpoint f(a) is included, not f(b))
#

    fs[0:n] = dft.ifft(fcoefs[0:n]) * n / ell
###    fs[0:n] = np.real(fff[0:n])

    return()

    for i in range(0,n):
        fs[i] = 0
        for j in range(0,n):
            fs[i] = fs[i] + fcoefs[j]*np.exp(2*np.pi*1j*j*i/n) / ell

    return()

#
#
#

def prodspin_polar_fft(fps,gps,nr,na,r,prods):
#
#    Evaluates inner products between f and rotations of g
#    using samples of f and g on an na-by-nr polar grid.
#

    ppz = np.empty(na,dtype='complex128')
    wwz = np.empty(na,dtype='complex128')
    rs = np.empty(nr+1,dtype='float64')
    rs2 = np.empty(nr+1,dtype='float64')
#
#
    fgprodz = np.empty([nr,na],dtype='complex128')
    fzs = np.empty([nr,na],dtype='complex128')
    gzs = np.empty([nr,na],dtype='complex128')

    kfs = np.empty(na,dtype='int32')


    dpi = np.pi
    da = 2*np.pi / na
    dr = r / nr

#
#    inner products in theta, for each r and rotation
#
    ell = 2*dpi

###    wp.prin2('fs=',fps,na)
###    wp.prin2('gs=',gps,na)

    dzero = 0.0
#
#    fourier coefficients
#

    fcoefs_fast(fps,nr,na,ell,fzs,kfs)
    fcoefs_fast(gps,nr,na,ell,gzs,kfs)
###    gk1d.ftrs_fast(fps,nr,na,dzero,ell,fzs,freqs,kfs)
###    gk1d.ftrs_fast(gps,nr,na,dzero,ell,gzs,freqs,kfs)

#
#    inner products
#
    fgprodz[0:nr,0:na] = fzs[0:nr,0:na] * np.conj(gzs[0:nr,0:na])

    rs[0:nr+1] = np.linspace(0,r,nr+1)
    wwz[0:na] = np.sum(fgprodz[0:nr,0:na] \
        * rs[0:nr,np.newaxis],axis=0) * dr

###    gk1d.iftr_fast(wwz,freqs,na,dzero,ell,prods)

    fsum_fast(wwz,na,ell,ppz)
    prods[0:na] = np.real(ppz[0:na])

    wp.prinz('fgprodz=',fgprodz,10)
    wp.prinz('wwz=',wwz,na)


    return()

    for j in range(0,na):
        wwz[j] = 0
        for i in range(0,nr):
            wwz[j] = wwz[j] + fgprodz[i,j] * rs[i] * dr

    gk1d.iftr_fast(wwz,freqs,na,dzero,ell,prods)


    return()

#
#
#

def evalmixn2d_polars(rs,angls,nr,na,cens,sigs,heis,ngaus,fs_pol):

    npts = nr*na

    xs = np.empty(npts,dtype='float64')
    ys = np.empty(npts,dtype='float64')

    pp_cos = np.empty([nr,na],dtype='float64')
    pp_sin = np.empty([nr,na],dtype='float64')

    fs_list = np.empty(npts,dtype='float64')

#
#    polar grid
#
    pp_cos[0:nr,0:na] = np.outer(rs[0:nr],np.cos(angls[0:na]))
    pp_sin[0:nr,0:na] = np.outer(rs[0:nr],np.sin(angls[0:na]))

    xs[0:npts] = np.reshape(pp_cos[0:nr,0:na],npts)
    ys[0:npts] = np.reshape(pp_sin[0:nr,0:na],npts)

    evalmixn2d_list(xs,ys,npts,cens,sigs,heis,ngaus,fs_list)
    fs_pol[0:nr,0:na] = np.reshape(fs_list[0:npts],[nr,na])




    return()

#
#
#

def prodspin_nufft(fs,gs,n,nr,na,r,prods):
#
#                            description:
#
#    Evaluates inner products between f and rotations of g from samples
#    of f and g on an n-by-n cartesian grid. The samples are interpolated
#    to a nr-by-na polar grid using the NUFFT. The inner product is computed
#    at for each rotation angle.
#
#
#                        input parameters:
#
#  fs,gs - n-by-n real arrays with samples of f, g on [-r,r) x [-r,r)
#  n - number of samples per dimensions; assumed to be an even integer (for now)
#  nr - number of samples in radial direction
#  na - number of samples in angular direction
#  r - length of each side
#
#
#                        output parameters:
#
#  prods - na-length real array containing the inner products between f
#    and each rotation of g
#


    xs = np.empty(na*nr,dtype='float64')
    ys = np.empty(na*nr,dtype='float64')
#
    pp_cos = np.empty([nr,na],dtype='float64')
    pp_sin = np.empty([nr,na],dtype='float64')
#
    fps_unwind = np.empty(nr*na,dtype='float64')
    gps_unwind = np.empty(nr*na,dtype='float64')
#
    fps = np.empty([nr,na],dtype='float64')
    gps = np.empty([nr,na],dtype='float64')
#
    angls = np.empty(na+1,dtype='float64')
    rs = np.empty(nr+1,dtype='float64')


    wp.prin2('fs, inside=',fs,n)
    wp.prin2('gs, inside=',gs,n)

    wp.prinf('n=',n,1)
    wp.prinf('nr=',nr,1)
    wp.prinf('na=',na,1)


###    for i in range(0,nr+1):
###        rs[i] = i*r / nr

    rs[0:nr+1] = np.linspace(0,r,nr+1)

    wp.prin2('rs=',rs,nr+1)
    wp.prin2('r=',r,1)


    dpi = np.pi

###    for i in range(0,na+1):
###        angls[i] = 2*dpi *i/na

    angls[0:na+1] = np.linspace(0,2*dpi,na+1)

    wp.prin2('angls=',angls,na+1)
    wp.prin2('2pi=',2*dpi,1)

#
#    polar grid
#
    npts = nr*na

    pp_cos[0:nr,0:na] = np.outer(rs[0:nr],np.cos(angls[0:na]))
    pp_sin[0:nr,0:na] = np.outer(rs[0:nr],np.sin(angls[0:na]))


    nh = np.int(n/2)

    deps = 1e-14

#
#    interpolate functions to polar grid using NUFFT
#
    xs[0:npts] = np.reshape(pp_cos[0:nr,0:na],npts) * np.pi / r
    ys[0:npts] = np.reshape(pp_sin[0:nr,0:na],npts) * np.pi / r

    nh = np.int(n/2)

    interp_nufft(fs,n,r,xs,ys,npts,fps_unwind,deps)
    fps[0:nr,0:na] = np.reshape(fps_unwind[0:npts],[nr,na])

    interp_nufft(gs,n,r,xs,ys,npts,gps_unwind,deps)
    gps[0:nr,0:na] = np.reshape(gps_unwind[0:npts],[nr,na])


    wp.prin2('gps=',gps,10)
    wp.prin2('fps=',fps,10)


#
#    evaluate all inner products
#
    prodspin_polar_fft(fps,gps,nr,na,r,prods)



    return()



    iftrs_2d_nudft(pts[:,0],pts[:,1],npts,fzs,n,r,freqs,fps_unwind_z)
    iftrs_2d_nudft(pts[:,0],pts[:,1],npts,gzs,n,r,freqs,gps_unwind_z)

    chk0 = np.sum(np.abs(fps_unwind2_z[0:npts] - fps_unwind_z[0:npts]))
    bbb = np.sum(np.abs(fps_unwind2_z[0:npts]))
    chk0 = chk0 / bbb
    wp.prin2('chk0=',chk0,1)


    chk0 = np.sum(np.abs(gps_unwind2_z[0:npts] - gps_unwind_z[0:npts]))
    bbb = np.sum(np.abs(gps_unwind2_z[0:npts]))
    chk0 = chk0 / bbb
    wp.prin2('chk0=',chk0,1)


    return()

#
#
#

def prodspin_cart(fs,gs,n,nr,na,r,prods):

    pp_cos = np.empty([nr,na],dtype='float64')
    pp_sin = np.empty([nr,na],dtype='float64')

    fps_unwind_z = np.empty(nr*na,dtype='complex128')
    gps_unwind_z = np.empty(nr*na,dtype='complex128')


    fps = np.empty([nr,na],dtype='float64')
    gps = np.empty([nr,na],dtype='float64')

    fzs = np.empty([n,n],dtype='complex128')
    gzs = np.empty([n,n],dtype='complex128')

    freqs = np.empty(n,dtype='float64')
    kfs = np.empty(n,dtype='int32')

    angls = np.empty(na+1,dtype='float64')
    rs = np.empty(nr+1,dtype='float64')

    pts = np.empty([na*nr,2],dtype='float64')


    wp.prin2('fs, inside=',fs,n)
    wp.prin2('gs, inside=',gs,n)

    wp.prinf('n=',n,1)
    wp.prinf('nr=',nr,1)
    wp.prinf('na=',na,1)


###    for i in range(0,nr+1):
###        rs[i] = i*r / nr

    rs[0:nr+1] = np.linspace(0,r,nr+1)

    wp.prin2('rs=',rs,nr+1)

    wp.prin2('r=',r,1)


    dpi = np.pi

###    for i in range(0,na+1):
###        angls[i] = 2*dpi *i/na

    angls[0:na+1] = np.linspace(0,2*dpi,na+1)

    wp.prin2('angls=',angls,na+1)
    wp.prin2('2pi=',2*dpi,1)


    dt = 2*r / n
    df = 1 / (2*r)

#
#    polar grid
#
    npts = nr*na

    pp_cos[0:nr,0:na] = np.outer(rs[0:nr],np.cos(angls[0:na]))
    pp_sin[0:nr,0:na] = np.outer(rs[0:nr],np.sin(angls[0:na]))

    pts[0:npts,0] = np.reshape(pp_cos[0:nr,0:na],npts)
    pts[0:npts,1] = np.reshape(pp_sin[0:nr,0:na],npts)



    nh = np.int(n/2)

#
#    interpolate functions to polar grid, stupidly
#
    ftrs_2d_fft(fs,n,r,fzs,freqs)
    iftrs_2d_nudft(pts[:,0],pts[:,1],npts,fzs,n,r,freqs,fps_unwind_z)
    fps[0:nr,0:na] = np.real(np.reshape(fps_unwind_z[0:npts],[nr,na]))


    ftrs_2d_fft(gs,n,r,gzs,freqs)
    iftrs_2d_nudft(pts[:,0],pts[:,1],npts,gzs,n,r,freqs,gps_unwind_z)
    gps[0:nr,0:na] = np.real(np.reshape(gps_unwind_z[0:npts],[nr,na]))


#
#    evaluate all inner products
#
    prodspin_polar_fft(fps,gps,nr,na,r,prods)



    return()


    wp.prinz('fps_unwind_z=',fps_unwind_z,1)


    wp.prin2('f center=',fs[nh,nh],1)

    ijk=6
    wp.prinz('fps_unwind_z=',fps_unwind_z[ijk*na:],na)
    wp.prin2('fps=',fps[ijk,:],na)
    wp.prin2('fps_true=',fps_true[ijk,:],na)


    wp.prinz('gps_unwind_z=',gps_unwind_z[ijk*na:],na)
    wp.prin2('gps=',gps[ijk,:],na)
    wp.prin2('gps_true=',gps_true[ijk,:],na)


    err = np.sum(np.abs(fps[0:nr,0:na] - fps_true[0:nr,0:na]))
    fsum = np.sum(np.abs(fps_true[0:nr,0:na]))
    err = err / fsum
    wp.prin2('err=',err,1)


    err = np.sum(np.abs(gps[0:nr,0:na] - gps_true[0:nr,0:na]))
    gsum = np.sum(np.abs(gps_true[0:nr,0:na]))
    err = err / gsum
    wp.prin2('err=',err,1)



    return()

#
#
#

def iftrs_2d_nudft(xs,ys,m,fzs,n,r,freqs,fs):
#
#    evaluates 2D IFFT at m arbitrary points (x_i,y_i),
#    stupidly (not using NUFFT). Does NOT assume function is
#    real (returns complex values, unless output array specified
#    by user is declared as real).
#
    zffs_x = np.empty([m,n],dtype='complex128')
    zffs_y = np.empty([m,n],dtype='complex128')

    dpi=np.pi
    zi = 1j

    wp.prin2('dpi=',dpi,1)


    dfreq = 1/(2*r)
    wp.prin2('dfreq=',dfreq,1)


###    zffs_x[0:m,0:n] = np.outer.mult(xs[0:m,np.newaxis], freqs[np.newaxis,0:n])
    zffs_x[0:m,0:n] = np.outer(xs[0:m], freqs[0:n])
    zffs_x[0:m,0:n] = np.exp(2 * dpi * zi * zffs_x[0:m,0:n])

    zffs_y[0:m,0:n] = np.outer(ys[0:m], freqs[0:n])
    zffs_y[0:m,0:n] = np.exp(2 * dpi * zi * zffs_y[0:m,0:n])

###    for i in range(0,m):
###        fs[i] = np.sum(zffs_x[i,0:n,np.newaxis] * fzs[0:n,0:n] * zffs_y[i,0:n])

###    fs[0:m] = np.real(\
###        np.sum(np.matmul(zffs_x[0:m,0:n], fzs[0:n,0:n]) * zffs_y[0:m,0:n], axis=1))

    fs[0:m] = np.sum(np.matmul(zffs_x[0:m,0:n], fzs[0:n,0:n]) * zffs_y[0:m,0:n], axis=1)
    fs[0:m] = fs[0:m] * dfreq**2

###    wp.prinz('fzs=',fzs[:,0:n],n*n)


    wp.prinz('fs, inside=',fs,m)

    return()

#
#
#

def prodstack1d_fft(fs,gs,m,n,ell,fgprods):

    fgprodz = np.empty([m,n],dtype='complex128')
    fzs = np.empty([m,n],dtype='complex128')
    gzs = np.empty([m,n],dtype='complex128')

    freqs = np.empty(n,dtype='float64')
    kfs = np.empty(n,dtype='int32')


    wp.prin2('fs=',fs,n)
    wp.prin2('gs=',gs,n)

###    exit()

    dzero = 0.0
#
#    fourier transforms
#
    gk1d.ftrs_fast(fs,m,n,dzero,ell,fzs,freqs,kfs)
    gk1d.ftrs_fast(gs,m,n,dzero,ell,gzs,freqs,kfs)

#
#    inner products
#
    fgprodz[0:m,0:n] = fzs[0:m,0:n] * np.conj(gzs[0:m,0:n])
    gk1d.iftrs_fast(fgprodz,freqs,m,n,dzero,ell,fgprods)


    wp.prinz('fgprodz=',fgprodz,10)


###    wp.prinz('fpadz=',fz,10)
###    wp.prinz('gpadz=',gz,10)



    return()

#
#
#

def prods1d_fft(fs,gs,n,r,fgprods):


    dt = 2*r / n

    fgprodz = np.empty(n,dtype='complex128')
    fz = np.empty(n,dtype='complex128')
    gz = np.empty(n,dtype='complex128')

    freqs = np.empty(n,dtype='float64')
    kfs = np.empty(n,dtype='int32')



###    wp.prini(13,0)


#
#    zero pad vectors
#
    wp.prin2('fs=',fs,n)
    wp.prin2('gs=',gs,n)

###    exit()




#
#    fourier transforms
#
    r2 = 2*r
    gk1d.ftr_fast(fs,n,0,r2,fz,freqs,kfs)
    gk1d.ftr_fast(gs,n,0,r2,gz,freqs,kfs)



#
#    inner products
#
    fgprodz[0:n] = fz[0:n] * np.conj(gz[0:n])
    gk1d.iftr_fast(fgprodz,freqs,n,0,r2,fgprods)


    wp.prinz('fgprodz=',fgprodz,10)
    wp.prinz('fpadz=',fz,10)
    wp.prinz('gpadz=',gz,10)



    return()

#
#
#

def prodspin_polar_dumb(fps,gps,nr,na,r,prods):

    gps2 = np.empty([nr,na],dtype='float64')

    wp.prinr2('fps=',fps,nr,na)


#
#    rotate, take inner product
#
    for k in range(0,na):
        cyclrots(gps,nr,na,k,gps2)
        prods[k] = prodpolar(fps,gps2,r,nr,na)


    wp.prinr2('gps=',gps,nr,na)
    wp.prinr2('gps2=',gps2,nr,na)

    wp.prin2('prods=',prods,na)


    pp = prodpolar(fps,fps,r,nr,na)
    pp2 = prodpolar(gps,gps,r,nr,na)

    wp.prin2('prods=',prods,na)
    wp.prin2('pp=',pp,1)
    wp.prin2('pp2=',pp2,1)


    return()

#
#
#

def cyclrots(x,m,n,k,x2):

    if (k < 0):
        k = n+k
###    wp.prinf('k, in cyclrot=',k,1)

    x2[0:m,k:n] = x[0:m,0:n-k]
    x2[0:m,0:k] = x[0:m,n-k:n]


###    wp.prin2('x=',x[1,:],n)
###    wp.prin2('x2=',x2[1,:],n)

    return()

    for j in range(0,m):

        for i in range(k,n):
            x2[j,i] = x[j,i-k]
    
        for i in range(0,k):
            x2[j,i] = x[j,n-k+i]


    return()

#
#
#

def rotvecs_dumb(xs,m,theta,ys):
#
#    rotates vectors by by theta
#

    cc = np.cos(theta)
    ss = np.sin(theta)


    for i in range(0,m):
        ys[i,0] = cc*xs[i,0] + ss*xs[i,1]
        ys[i,1] = - ss*xs[i,0] + cc*xs[i,1]


    return()

#
#
#
####################################################################################
#
#            This is the end of the test code and the beginning of the
#            multiscale Gaussian metric code proper.
#
####################################################################################
#
#
#            This file contains the following user-callable functions:
#
#  gk2d_pad_fast - evaluates multiscale distance, with zero-padding to ensure
#        the domain can hold the convolution with the widest Gaussian.
#  gk2d_nopad_fast - evaluates multiscale distance, without zero-padding.
#  gk2d_rots_pad - evaluates multiscale distances between f and rotations of g,
#        with zero-padding to ensure the domain can hold the convolution with
#        the widest Gaussian.
#  gk2d_rots - evaluates multiscale distances between f and rotations of g,
#        without zero-padding.
#  eucl_rots - evaluates L2 distances between f and rotations of g, from
#        Cartesian samples.
#
#

def gk2d_pad_fast(gs,hs,n,r,p,alf,tau0,omg,nlevs,deps):
#
#                            description:
#
#    Evaluates the multiscale distance between g and h from samples of g and h
#    on an n-by-n cartesian grid. The inputs are first zero-padded,
#    and the radius extended, to hold the convolutions.
#
#
#                        input parameters:
#
#  fs,gs - n-by-n real arrays with samples of g, h on [-r,r) x [-r,r)
#  n - number of samples per dimensions; assumed to be an even integer (for now)
#  r - length of each side in real space
#  p - power of metric
#  alf,tau0,omg - scale parameters; weight at level k is tau0 * omg**(k*alf/2)
#  nlevs - number of levels
#  deps - precision used to evaluate extra number of samples/extended radius
#
#
#                        return parameters:
#
#  dist - multiscale distance between g and h
#
#

###    [n2,nadd,r2] = gk1d.extlength_pad(n,tau0,deps,r)
    [n2,nadd,r2] = extlength_pad(n,tau0,deps,r)

    gpad = np.empty([n2,n2],dtype='float64')
    hpad = np.empty([n2,n2],dtype='float64')

    wp.prinf('n2=',n2,1)
    wp.prinf('nadd=',nadd,1)
    wp.prin2('r2=',r2,1)
    wp.prin2('r=',r,1)


    padzeros2d(gs,n,gpad,nadd)
    padzeros2d(hs,n,hpad,nadd)


    dnorm = gk2d_nopad_fast(gpad,hpad,n2,r2,p,alf,tau0,omg,nlevs)


    return(dnorm)

#
#
#

def eucl_rots(fs,gs,n,nr,na,na2,r,rf,dists):
#
#                            description:
#
#    Evaluates L2 distances between f and rotations of g from samples
#    of f and g on an n-by-n cartesian grid. The samples are interpolated
#    to a nr-by-na polar grid using the NUFFT. Inner products are computed
#    at na2 equispaced angles. Norms are computed on the polar grid.
#
#
#                        input parameters:
#
#  fs,gs - n-by-n real arrays with samples of f, g on [-r,r) x [-r,r)
#  n - number of samples per dimensions; assumed to be an even integer (for now)
#  nr - number of samples in radial direction
#  na - number of samples in angular direction
#  na2 - number of inner products; assumed na2 >= na
#  r - length of each side in real space
#  rf - maximum frequency magnitude
#
#
#                        output parameters:
#
#  dists - na2-length real array containing the distances between f
#    and each rotation of g
#
#
#                        return parameters:
#
#  dist_min - minimum distance, i.e. minimum of the array dists
#


    na = 2*np.int((na+1)/2)

    npts = nr*na2

    prods = np.empty(npts,dtype='float64')



    dt = 2*r / n

#
#    all inner products, and squared norms
#

###    wp.prini(0,0)
    [fsq,gsq] = prods_rots(fs,gs,n,nr,na,na2,r,rf,prods)
###    wp.prini(13,0)
#

#
#    distances, and minimum distance
#
    dists[0:na2] = (fsq + gsq - 2*prods[0:na2])
    dists[0:na2] = np.sqrt(np.abs(dists[0:na2]))

    dist_min = np.min(dists[0:na2])



###    wp.prin2('prods=',prods,na2)
###    wp.prin2('dists, before=',dists,na2)

###    return(dist_min)


    wp.prin2('fsq=',fsq,1)
    wp.prin2('gsq=',gsq,1)

    fsq2 = np.sum(fs[0:n,0:n]**2) * dt**2
    gsq2 = np.sum(gs[0:n,0:n]**2) * dt**2
    wp.prin2('fsq2=',fsq2,1)
    wp.prin2('gsq2=',gsq2,1)

###    wp.prin2('fsq=',np.sqrt(fsq),1)

    ratio = fsq / fsq2
    wp.prin2('ratio=',ratio,1)

    ratio = gsq / gsq2
    wp.prin2('ratio=',ratio,1)


###    exit()


    return(dist_min)

#
#
#

def prods_rots(fs,gs,n,nr,na,na2,r,rf,prods):
#
#                            description:
#
#    Evaluates inner products between f and rotations of g from samples
#    of f and g on an n-by-n cartesian grid. The samples are used to evaluate
#    the Fourier coefficients of f and g on an nr-by-na polar grid, using the
#    NUFFT. Inner products between the Fourier transforms are then computed at
#    na2 equispaced angles.
#
#
#                        input parameters:
#
#  fs,gs - n-by-n real arrays with samples of f, g on [-r,r) x [-r,r)
#  n - number of samples per dimensions; assumed to be an even integer (for now)
#  nr - number of samples in radial direction
#  na - number of samples in angular direction
#  na2 - number of inner products; assumed na2 >= na
#  r - length of each side in real space
#  rf - maximum frequency magnitude
#
#
#                        output parameters:
#
#  prods - na2-length real array containing the inner products between
#    and each rotation of g
#
#
#                        return parameters:
#
#  fsq,gsq - squared norms of f and g, evaluated from polar grid
#

    fzpols_unwind = np.empty(na*nr,dtype='complex128')
    fzpols = np.empty([nr,na],dtype='complex128')
#
    gzpols_unwind = np.empty(na*nr,dtype='complex128')
    gzpols = np.empty([nr,na],dtype='complex128')
#
    czs = np.empty(na*nr,dtype='complex128')
#
    xs = np.empty(na*nr,dtype='float64')
    ys = np.empty(na*nr,dtype='float64')

#
    pp_cos = np.empty([nr,na],dtype='float64')
    pp_sin = np.empty([nr,na],dtype='float64')
#
    fs_comp = np.empty([n,n],dtype='complex128')
    gs_comp = np.empty([n,n],dtype='complex128')
#
    angls = np.empty(na+1,dtype='float64')
    rfs = np.empty(nr+1,dtype='float64')
###    rs = np.empty(nr+1,dtype='float64')



    wp.prin2('r=',r,1)
    wp.prin2('rf=',rf,1)

    wp.prin2('fs, inside=',fs,n)
    wp.prin2('gs, inside=',gs,n)

    wp.prinf('n=',n,1)
    wp.prinf('nr=',nr,1)
    wp.prinf('na=',na,1)



###    for i in range(0,nr+1):
###        rs[i] = i*r / nr

    rfs[0:nr+1] = np.linspace(0,rf,nr+1)

    wp.prin2('rfs=',rfs,nr+1)
    wp.prin2('rf=',rf,1)


    dpi = np.pi

###    for i in range(0,na+1):
###        angls[i] = 2*dpi *i/na

    angls[0:na+1] = np.linspace(0,2*dpi,na+1)

    wp.prin2('angls=',angls,na+1)
    wp.prin2('2pi=',2*dpi,1)


#
#    polar grid
#
    npts = nr*na

    pp_cos[0:nr,0:na] = np.outer(rfs[0:nr],np.cos(angls[0:na]))
    pp_sin[0:nr,0:na] = np.outer(rfs[0:nr],np.sin(angls[0:na]))


    nh = np.int(n/2)

    xs[0:npts] = np.reshape(pp_cos[0:nr,0:na],npts) * dpi * 4*r / n
    ys[0:npts] = np.reshape(pp_sin[0:nr,0:na],npts) * dpi * 4*r / n


    wp.prin2('pp_cos=',pp_cos[:,0],nr)
    wp.prin2('rf=',rf,1)


    wp.prin2('xs=',xs,10)
    wp.prin2('ys=',ys,10)

###    exit()


    wp.prin2('dpi=',dpi,1)

    dt = 2*r / n


#
#    evaluate Fourier transforms on polar grid
#
    fs_comp[0:n,0:n] = fs[0:n,0:n]
    fi.nufft2d2(xs, ys, fs_comp, fzpols_unwind, eps=1e-15, isign=-1)
    fzpols_unwind[0:npts] = fzpols_unwind[0:npts]*dt**2


    gs_comp[0:n,0:n] = gs[0:n,0:n]
    fi.nufft2d2(xs, ys, gs_comp, gzpols_unwind, eps=1e-15, isign=-1)
    gzpols_unwind[0:npts] = gzpols_unwind[0:npts]*dt**2



    wp.prin2('fs=',fs,10)
    wp.prinz('fs_comp=',fs_comp,10)
    wp.prinz('czs=',fzpols_unwind,na+4)


    wp.prinf('npts=',npts,1)

    wp.prinf('n=',n,1)
    wp.prinf('na=',na,1)
    wp.prinf('nr=',nr,1)


    fzpols[0:nr,0:na] = np.reshape(fzpols_unwind[0:npts],[nr,na])
    gzpols[0:nr,0:na] = np.reshape(gzpols_unwind[0:npts],[nr,na])


#
#    evaluate all inner products
#

###    wp.prinz('fzpols=',fzpols[5,:],12)
###    wp.prinz('gzpols=',gzpols[5,:],12)


    err_z = prodspin_polar(fzpols,gzpols,nr,na,na2,rf,prods)

    wp.prin2('prods, in=',prods,na2)


#
#    compute squared norms using polar grid
#

    dr = rf / nr
    da = 2*dpi / na

    fsq = np.sum(np.abs(fzpols[0:nr,0:na])**2 * rfs[0:nr,np.newaxis])
    fsq = fsq * dr * da

    gsq = np.sum(np.abs(gzpols[0:nr,0:na])**2 * rfs[0:nr,np.newaxis])
    gsq = gsq * dr * da



    return(fsq,gsq)

#
#
#

def prodspin_polar(fps,gps,nr,na,na2,r,prods_pad):
#
#                            description:
#
#    Evaluates inner products between f and rotations of g
#    using samples of f and g on an na-by-nr polar grid.
#    Inner products are computed at na2 equispaced angles.
#    It is assumed that the input functions are either real-valued
#    or the Fourier transforms of real-valued functions, hence
#    that the inner products are real.
#
#
#                        input parameters:
#
#  fps,gps - nr-by-na real or complex arrays with samples of f,g
#    on a polar grid
#  nr - number of samples in radial direction
#  na - number of samples in angular direction; assumed to be even
#  na2 - number of inner products; assumed that na2 >= na
#  r - radius of support
#
#
#                        output parameters:
#
#  prods_pad - na2-length *real* array of inner products
#
#
#                        return parameters:
#
#  err_z - average magnitude of the the complex parts of the inner
#    products; it should be on the order of machine precision
#

    wwz_pad = np.empty(na2,dtype='complex128')
    wwz = np.empty(na,dtype='complex128')
    rs = np.empty(nr+1,dtype='float64')
#
    fgprodz = np.empty([nr,na],dtype='complex128')
    fzs = np.empty([nr,na],dtype='complex128')
    gzs = np.empty([nr,na],dtype='complex128')
#
    kfs = np.empty(na,dtype='int32')
    kfs_pad = np.empty(na2,dtype='int32')
#
    ppz = np.empty(na,dtype='complex128')
    ppz_pad = np.empty(na2,dtype='complex128')
    prods = np.empty(na,dtype='float64')


    dpi = np.pi
    da = 2*np.pi / na
    dr = r / nr

#
#    inner products in theta, for each r and rotation
#
###    wp.prinz('fs=',fps[10,:],na)
###    wp.prinz('gs=',gps[10,:],na)

    dzero = 0.0
#
#    fourier transforms
#
    fzs[0:nr,0:na] = dft.fft(fps[0:nr,0:na]) * da
    gzs[0:nr,0:na] = dft.fft(gps[0:nr,0:na]) * da

###    fcoefs_fast(fps,nr,na,ell,fzs,kfs)
###    fcoefs_fast(gps,nr,na,ell,gzs,kfs)



#
#    Fourier coefficients of inner products function
#
    fgprodz[0:nr,0:na] = fzs[0:nr,0:na] * np.conj(gzs[0:nr,0:na])

    rs[0:nr+1] = np.linspace(0,r,nr+1)
    wwz[0:na] = np.sum(fgprodz[0:nr,0:na] \
        * rs[0:nr,np.newaxis],axis=0) * dr


#
#    on-grid inner products
#
###    fsum_fast(wwz,na,ell,ppz)
    ppz[0:na] = dft.ifft(wwz[0:na]) / da

    prods[0:na] = np.real(ppz[0:na])
    wp.prin2('prods=',prods,na)

###    exit()

    wp.prinz('fgprodz=',fgprodz,10)
    wp.prinz('wwz=',wwz,na)


###    wp.prinf('kfs=',kfs,na)

    na2_h = np.int(na2/2)

    kfs_pad[0:na2] = range(0,na2)
    kfs_pad[na2_h:na2] = kfs_pad[na2_h:na2] - na2

    wp.prinf('kfs_pad=',kfs_pad,na2)
###    wp.prinf('kfs=',kfs,na)



#
#    zero pad Fourier coefficents
#
    na_h = np.int(na/2)
    wwz_pad[0:na2] = 0
    wwz_pad[0:na_h] = wwz[0:na_h]
    wwz_pad[na2-na_h:na2] = wwz[na_h:na]




#
#    off-grid inner products
#
    da2 = 2*np.pi / na2
###    fsum_fast(wwz_pad,na2,ell,ppz_pad)
    ppz_pad[0:na2] = dft.ifft(wwz_pad[0:na2]) / da2

    prods_pad[0:na2] = np.real(ppz_pad[0:na2])


    err_z = np.sum(np.abs(np.imag(ppz_pad[0:na2]))) / na2

    wp.prin2('err_z=',err_z,1)


    return(err_z)

#
#
#

def extlength_pad(n,tau0,deps,r):

    dt = 2*r/n

#
#    determine extended length
#
###    r_add = np.sqrt(tau0*np.log(1/deps/np.sqrt(tau0)))
    r_add = np.sqrt(tau0*np.log(1/deps))
#
    n_add = r_add / dt
    n_add = np.max([np.int(n_add),0])
#
    n_ext = n+2*n_add
    r_ext = r + n_add*dt


###    wp.prin2('r_add=',r_add,1)
###    wp.prin2('r_add2=',r_add2,1)


    return(n_ext,n_add,r_ext)

#
#
#

def padzeros2d(fs,n,fpad,nplus):

    npad = n+2*nplus
    fpad[0:npad,0:npad] = 0
    fpad[nplus:nplus+n,nplus:nplus+n] = fs[0:n,0:n]

###    wp.prinr2('fs=',fs,2,6)
###    wp.prinr2('fpad=',fpad[:,npad-2*nplus:],nplus+2,6)

    return()

#
#
#

def gk2d_nopad_fast(gs,hs,n,r,p,alf,tau0,omg,nlevs):
#
#                            description:
#
#    Evaluates the multiscale distance between g and h from samples of
#    g and h on an n-by-n cartesian grid. The inputs are NOT zero-padded.
#
#
#                        input parameters:
#
#  fs,gs - n-by-n real arrays with samples of g, h on [-r,r) x [-r,r)
#  n - number of samples per dimensions; assumed to be an even integer (for now)
#  r - length of each side in real space
#  p - power of metric
#  alf,tau0,omg - scale parameters; weight at level k is tau0 * omg**(k*alf/2)
#  nlevs - number of levels
#
#
#                        return parameters:
#
#  dist - multiscale distance between g and h
#
#


    fs = np.empty([n,n],dtype='float64')
    fstack = np.empty([n,n,nlevs],dtype='float64')
    fstackz = np.empty([n,n,nlevs],dtype='complex128')

    fzs = np.empty([n,n],dtype='complex128')

    ghats = np.empty([n,n,nlevs],dtype='float64')

    taus = np.empty(nlevs,dtype='float64')
    whts = np.empty(nlevs,dtype='float64')
    ds = np.empty(nlevs,dtype='float64')
#
    prods = np.empty(nlevs,dtype='float64')
    freqs = np.empty(n,dtype='float64')
    kfs = np.empty(n,dtype='int32')
#



#
#    differences
#
    fs[0:n,0:n] = gs[0:n,0:n] - hs[0:n,0:n]



#
#    frequencies, in DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)



    ftrs_2d_fft(fs,n,r,fzs,freqs)

    wp.prin2('freqs=',freqs,n)



    wp.prin2('r=',r,1)
    wp.prin2('tau0=',tau0,1)
    wp.prin2('alf=',alf,1)
    wp.prin2('p=',p,1)
    wp.prin2('omg=',omg,1)



    nh = np.int(n/2)

    wp.prin2('fs=',fs[nh,:],n)


#
#    all widths
#
    taus[0:nlevs] = tau0 * omg**np.linspace(0,nlevs-1,nlevs)

    wp.prin2('taus=',taus,nlevs)


#
#    all convolutions
#
    ghats_all(ghats,n,nlevs,taus,freqs)
    fstackz[0:n,0:n,0:nlevs] = fzs[0:n,0:n,np.newaxis] * ghats[0:n,0:n,0:nlevs]

###    for ijk in range(0,nlevs):
###        iftr_2d_fft(fstackz[0:n,0:n,ijk],n,r,fstack[0:n,0:n,ijk],freqs)

    iftrs_2d_fft(fstackz[0:n,0:n,0:nlevs],n,nlevs,r,fstack,freqs)


    wp.prin2('fstack=',fstack[nh,:,3],n)


###    conv_gaus2d(fs,n,r,taus[0],fconvs)


    dt = 2*r / n
    wp.prin2('dt=',dt,1)


    ds[0:nlevs] = np.sum(np.sum(fstack[0:n,0:n,0:nlevs]**2,axis=1),axis=0) * dt**2
    ds[0:nlevs] = np.sqrt(ds[0:nlevs])

    wp.prin2('ds=',ds,nlevs)



    whts[0:nlevs] = taus[0:nlevs]**(alf/2)
    dist = np.sum((ds[0:nlevs] * whts[0:nlevs])**p)

    dist = dist**(1/p)



    prods[0:nlevs] = whts[0:nlevs] * ds[0:nlevs]

    wp.prin2('prods=',prods,nlevs)
    wp.prin2('dist=',dist,1)


    return(dist)

#
#
#

def ftrs_2d_fft(fs,n,r,fzs,freqs):

    zzzs = np.empty([n,n],dtype='complex128')
    kfs = np.empty(n,dtype='int32')

    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

    dt = 2*r / n

    zzzs[0:n,0:n] = np.outer(np.exp(2*np.pi*1j*r*freqs[0:n]), \
        np.exp(2*np.pi*1j*r*freqs[0:n]))

    fzs[0:n,0:n] = dft.fft2(fs[0:n,0:n]) * dt**2
    fzs[0:n,0:n] = zzzs[0:n,0:n] * fzs[0:n,0:n]



###    wp.prinz('fzs=',fzs[:,0:n],n*n)



###    ftrs_2d_dumb(fs,n,r,fzs2,freqs2)
###    wp.prinz('fzs2=',fzs2[:,0:n],n*n)

    return()

#
#
#

def ghats_all(ghats,n,nlevs,taus,freqs):

    sigs = np.empty(nlevs,dtype='float64')
    ffs0 = np.empty([n,nlevs],dtype='float64')


    sigs[0:nlevs] = 1 / (np.pi**2 * taus[0:nlevs])


    ffs0[0:n,0:nlevs] = np.exp(-np.divide(freqs[0:n,np.newaxis]**2, \
        sigs[np.newaxis,0:nlevs]))

    ghats[0:n,0:n,0:nlevs] = np.multiply(ffs0[0:n,np.newaxis,0:nlevs], \
        ffs0[np.newaxis,0:n,0:nlevs])


    return()

#
#
#

def iftrs_2d_fft(fzs,n,m,r,fs,freqs):
    zzzs = np.empty([n,n],dtype='complex128')
    gggs = np.empty([n,n,m],dtype='complex128')
    fff = np.empty([n,n,m],dtype='complex128')

###    hzh = np.empty([n,n],dtype='float64')

    dpi=np.pi
    zi = 1j

    wp.prin2('dpi=',dpi,1)


    dfreq = 1/(2*r)
    wp.prin2('dfreq=',dfreq,1)


    zzzs[0:n,0:n] = np.outer(np.exp(-2*dpi*zi*freqs[0:n]*r), \
        np.exp(-2*dpi*zi*freqs[0:n]*r))
    gggs[0:n,0:n,0:m] = np.multiply(fzs[0:n,0:n,0:m],zzzs[0:n,0:n,np.newaxis])


#
#    inverse DFT, rescaled
#
    fff[0:n,0:n,0:m] = dft.ifft2(gggs[0:n,0:n,0:m],axes=[0,1]) * dfreq**2 * n**2
    fs[0:n,0:n,0:m] = np.real(fff[0:n,0:n,0:m])


    return()


    ijk=0
    hzh[0:n,0:n] = np.real(dft.ifft2(gggs[0:n,0:n,ijk]) * dfreq**2 * n**2)



    wp.prin2('hzh=',hzh,12)
    wp.prin2('fs=',fs[:,0:n,ijk],12)


###    exit()

    return()

#
#
#

def gk2d_rots_pad(fs,gs,n,nr,na,na2,r,rf,p,alf,tau0,omg,nlevs,deps,gkdists):
#
#                            description:
#
#    Evaluates all multiscale distances between f and rotations of g
#    from samples of f and g on an n-by-n cartesian grid. The samples are
#    interpolated to a nr-by-na polar grid using the NUFFT. Distances are
#    computed at na2 equispaced angles. The inputs are first zero-padded,
#    and the radius extended, to hold the convolutions.
#
#
#                        input parameters:
#
#  fs,gs - n-by-n real arrays with samples of f, g on [-r,r) x [-r,r)
#  n - number of samples per dimensions; assumed to be an even integer (for now)
#  nr - number of samples to interpolate in radial direction
#  na - number of samples to interpolate in angular direction
#  na2 - number of distances; assumed to be at least as big as na
#  r - length of each side in real space
#  rf - maximum frequency magnitude
#  p - power of metric
#  alf,tau0,omg - scale parameters; weight at level k is tau0 * omg**(k*alf/2)
#  nlevs - number of levels
#  deps - precision used to evaluate extra number of samples/extended radius
#
#
#                        output parameters:
#
#  gkdists - na2-length real array containing the distances at each rotation
#    angle
#
#
#                        return parameters:
#
#  dist_min - minimum distance, i.e. minimum of gkdists
#
#

    [n_ext,nadd,r_ext] = extlength_pad(n,tau0,deps,r)

    fpad = np.empty([n_ext,n_ext],dtype='float64')
    gpad = np.empty([n_ext,n_ext],dtype='float64')


###    wp.prini(13,0)
    wp.prinf('n=',n,1)
    wp.prinf('n_ext=',n_ext,1)
    wp.prinf('nadd=',nadd,1)
    wp.prin2('r_ext=',r_ext,1)
    wp.prin2('r=',r,1)
###    wp.prini(0,0)

###    exit()

    padzeros2d(fs,n,fpad,nadd)
    padzeros2d(gs,n,gpad,nadd)

    wp.prinr2('fs=',fs[:,n-6:],n,6)
    wp.prinr2('fpad=',fpad[:,n_ext-2*nadd:],n_ext,6)



    wp.prinr2('fs=',fs,n,6)
    wp.prinr2('fpad=',fpad,n_ext,6)

    dist_min = gk2d_rots(fpad,gpad,n_ext,nr,na,na2,r_ext,rf, \
        p,alf,tau0,omg,nlevs,gkdists)

    return(dist_min)

#
#
#

def extlength_pad(n,tau0,deps,r):

    dt = 2*r/n

#
#    determine extended length
#
###    r_add = np.sqrt(tau0*np.log(1/deps/np.sqrt(tau0)))
    r_add = np.sqrt(tau0*np.log(1/deps))
#
    n_add = r_add / dt
    n_add = np.max([np.int(n_add),0])
#
    n_ext = n+2*n_add
    r_ext = r + n_add*dt


###    wp.prin2('r_add=',r_add,1)
###    wp.prin2('r_add2=',r_add2,1)


    return(n_ext,n_add,r_ext)

#
#
#

def gk2d_rots(fs,gs,n,nr,na,na2,r,rf,p,alf,tau0,omg,nlevs,gkdists):
#
#                            description:
#
#    Evaluates all multiscale distances between f and rotations of g
#    from samples of f and g on an n-by-n cartesian grid. The samples are
#    interpolated to a nr-by-na polar grid using the NUFFT. Distances are
#    computed at na2 equispaced angles. The inputs are NOT zero-padded.
#
#
#                        input parameters:
#
#  fs,gs - n-by-n real arrays with samples of f, g on [-r,r) x [-r,r)
#  n - number of samples per dimensions; assumed to be an even integer (for now)
#  nr - number of samples to interpolate in radial direction
#  na - number of samples to interpolate in angular direction; if it is odd,
#    it is rounded up to an even number
#  na2 - number of distances; assumed to be at least as big as na
#  r - length of each side in real space
#  rf - maximum frequency magnitude
#  p - power of metric
#  alf,tau0,omg - scale parameters; weight at level k is tau0 * omg**(k*alf/2)
#  nlevs - number of levels
#
#
#                        output parameters:
#
#  gkdists - na2-length real array containing the distances at each rotation
#    angle
#
#
#                        return parameters:
#
#  dist_min - minimum distance, i.e. minimum of gkdists
#
#


    wp.prinf('na, before=',na,1)


    na = 2*np.int((na+1)/2)

    wp.prinf('na=',na,1)


    fzs = np.empty([nr,na],dtype='complex128')
    gzs = np.empty([nr,na],dtype='complex128')
#
    kfs = np.empty(na,dtype='int32')
    kfs_pad = np.empty(na2,dtype='int32')
#
    xs = np.empty(na*nr,dtype='float64')
    ys = np.empty(na*nr,dtype='float64')
#
    fsqs = np.empty(nlevs,dtype='float64')
    gsqs = np.empty(nlevs,dtype='float64')
#
    fzpols = np.empty([nr,na],dtype='complex128')
    gzpols = np.empty([nr,na],dtype='complex128')
#
    taus = np.empty(nlevs,dtype='float64')
    whts = np.empty(nlevs,dtype='float64')
#
    prods = np.empty([nlevs,na2],dtype='float64')
    dists = np.empty([nlevs,na2],dtype='float64')


    ghats_pol = np.empty([nr,nlevs],dtype='float64')

    angls = np.empty(na+1,dtype='float64')
    rfs = np.empty(nr+1,dtype='float64')
#
    wwzs = np.empty([nlevs,na],dtype='complex128')
    fgprodz = np.empty([nr,na],dtype='complex128')

#


###    gkdists = np.empty(na2,dtype='float64')

    wp.prin2('alf, inside=',alf,1)

    wp.prinf('n=',n,1)
    wp.prinf('na=',na,1)
    wp.prinf('na2=',na2,1)

    wp.prin2('tau0=',tau0,1)
    wp.prinf('nlevs=',nlevs,1)


    wp.prin2('r=',r,1)
    wp.prin2('rf=',rf,1)


    wp.prin2('fs=',fs,10)
    wp.prin2('gs=',gs,10)


    dpi = np.pi

    da = 2*np.pi / na
    dr = rf / nr
    dt = 2*r / n

    dzero = 0.0
    ell = 2*dpi

    npts = nr*na

#
#    polar grid
#
###    nh = np.int(n/2)
    polar_grid(rf,nr,na,xs,ys,rfs,angls)


#
#    Fourier transforms of f and g on polar grid
#

    wp.prinf('n=',n,1)
    wp.prinf('nr=',nr,1)

    fourier_polar_nufft(fs,r,n,nr,na,xs,ys,fzpols)
    fourier_polar_nufft(gs,r,n,nr,na,xs,ys,gzpols)


###    wp.prinz('gzpols=',gzpols[:,0:na],10*na)
###    wp.prinz('fzpols=',fzpols[:,0:na],10*na)


#
#    all widths
#
    taus[0:nlevs] = tau0 * omg**np.linspace(0,nlevs-1,nlevs)

    wp.prin2('taus=',taus,nlevs)

###    exit()


#
#    gaussian fourier transforms on polar grid
#
    ghats_polar(rfs,nr,taus,nlevs,ghats_pol)


###    fcoefs_fast(fzpols,nr,na,ell,fzs,kfs)
###    fcoefs_fast(gzpols,nr,na,ell,gzs,kfs)
    fzs[0:nr,0:na] = dft.fft(fzpols[0:nr,0:na]) * da
    gzs[0:nr,0:na] = dft.fft(gzpols[0:nr,0:na]) * da



###    wp.prinz('fzpols=',fzpols[40,:],na)
###    wp.prinz('fzs=',fzs[40,:],na)

    wp.prinf('na=',na,1)

###    exit()

    fgprodz[0:nr,0:na] = fzs[0:nr,0:na] * np.conj(gzs[0:nr,0:na])

###    wp.prinz('fgprodz=',fgprodz[40,:],na)
###    wp.prinf('na=',na,1)


#
#    rotated inner products, and fourier coefficients
#
    wwzs[0:nlevs,0:na] = np.sum(fgprodz[0:nr,np.newaxis,0:na] \
        * ghats_pol[0:nr,0:nlevs,np.newaxis]**2 \
        * rfs[0:nr,np.newaxis,np.newaxis],axis=0) * dr


###    wp.prinz('wwzs=',wwzs[2,:],na)
###    exit()

#
#    interpolate to finer grid
#
    interps1d_fft(wwzs,nlevs,na,na2,prods,kfs_pad)


    wp.prinz('wwzs=',wwzs[0:nlevs,0:na],na*nlevs)


#
#    squared norms at each level
#
    fsqs[0:nlevs] = np.sum(np.sum(np.abs(fzpols[0:nr,np.newaxis,0:na])**2 \
        * ghats_pol[0:nr,0:nlevs,np.newaxis]**2 \
        * rfs[0:nr,np.newaxis,np.newaxis],axis=0),axis=1) * dr * da

    gsqs[0:nlevs] = np.sum(np.sum(np.abs(gzpols[0:nr,np.newaxis,0:na])**2 \
        * ghats_pol[0:nr,0:nlevs,np.newaxis]**2 \
        * rfs[0:nr,np.newaxis,np.newaxis],axis=0),axis=1) * dr * da

#
#    rotated distances
#
    dists[0:nlevs,0:na2] = fsqs[0:nlevs,np.newaxis] + gsqs[0:nlevs,np.newaxis] \
         - 2*prods[0:nlevs,0:na2]
    dists[0:nlevs,0:na2] = np.sqrt(np.abs(dists[0:nlevs,0:na2]))



#
#    multiscale distances
#
    whts[0:nlevs] = taus[0:nlevs]**(alf/2)
    wp.prin2('whts=',whts,nlevs)


    gkdists[0:na2] = np.sum((dists[0:nlevs,0:na2] * whts[0:nlevs,np.newaxis])**p,axis=0)
    gkdists[0:na2] = gkdists[0:na2]**(1/p)


    dist_min = np.min(gkdists[0:na2])


    wp.prin2('dists=',dists[0,:],na2)
###    wp.prin2('dists=',dists[3,:],na2)
    wp.prin2('dists=',dists[nlevs-1,:],na2)




    return(dist_min)

#
#
#

def fourier_polar_nufft(fs,r,n,nr,na,xs,ys,fzpols):
#
#    evaluates Fourier transform on polar grid, from samples
#    of function on cartesian grid over [-r,r) x [-r,r);
#    assumes grid is preprocessed
#
    npts = nr*na

    xs2 = np.empty(npts,dtype='float64')
    ys2 = np.empty(npts,dtype='float64')
#
    fs_comp = np.empty([n,n],dtype='complex128')
    fzpols_unwind = np.empty(na*nr,dtype='complex128')


    dpi = np.pi

    xs2[0:npts] = xs[0:npts] * dpi * 4*r / n
    ys2[0:npts] = ys[0:npts] * dpi * 4*r / n

###    exit()




    fs_comp[0:n,0:n] = fs[0:n,0:n]


    dt = 2*r / n

    wp.prinf('nr=',nr,1)
    wp.prinf('na=',na,1)


    fi.nufft2d2(xs2, ys2, fs_comp, fzpols_unwind, eps=1e-15, isign=-1)
    fzpols_unwind[0:npts] = fzpols_unwind[0:npts]*dt**2
    fzpols[0:nr,0:na] = np.reshape(fzpols_unwind[0:npts],[nr,na])



    return()

#
#
#

def interps1d_fft(fhats,m,n,n2,fs,kfs_pad):
#
#    takes n fourier coefficients for each of m functions,
#    zero pads them, and evaluates each function at n2 samples
#
    fhats_pad = np.empty([m,n2],dtype='complex128')
    ppz_pad = np.empty([m,n2],dtype='complex128')


    dpi = np.pi
    dzero = 0.0
    ell = 2*dpi


    n2_h = np.int(n2/2)

    kfs_pad[0:n2] = range(0,n2)
    kfs_pad[n2_h:n2] = kfs_pad[n2_h:n2] - n2

#
#    zero pad fourier transform
#
    n_h = np.int(n/2)
    fhats_pad[0:m,0:n2] = 0
    fhats_pad[0:m,0:n_h] = fhats[0:m,0:n_h]
    fhats_pad[0:m,n2-n_h:n2] = fhats[0:m,n_h:n]


###    fsums_fast(fhats_pad,m,n2,ell,ppz_pad)
    ppz_pad[0:m,0:n2] = dft.ifft(fhats_pad[0:m,0:n2],axis=1) * n2 / ell
    fs[0:m,0:n2] = np.real(ppz_pad[0:m,0:n2])


    return()

#
#    . . . same result, looping over rows
#
    for i in range(0,m):
        interp1d_fft(fhats[i,:],n,n2,fs[i,:],kfs_pad)

    return()

#
#
#

def interp1d_fft(fhats,n,n2,fs,kfs_pad):
#
#    takes n fourier coefficients, zero pads them,
#    and evaluates f at n2 samples
#
    fhats_pad = np.empty(n2,dtype='complex128')
    ppz_pad = np.empty(n2,dtype='complex128')


    dpi = np.pi
    dzero = 0.0
    ell = 2*dpi


    n2_h = np.int(n2/2)

    kfs_pad[0:n2] = range(0,n2)
    kfs_pad[n2_h:n2] = kfs_pad[n2_h:n2] - n2

#
#    zero pad fourier transform
#
    n_h = np.int(n/2)
    fhats_pad[0:n2] = 0
    fhats_pad[0:n_h] = fhats[0:n_h]
    fhats_pad[n2-n_h:n2] = fhats[n_h:n]

    fsum_fast(fhats_pad,n2,ell,ppz_pad)
    fs[0:n2] = np.real(ppz_pad[0:n2])


    return()

#
#
#

def fcoefs_fast(fs,m,n,ell,fzs,kfs):
#
#                            description:
#
#    This code evaluates the Fourier coefficients of multiple functions f.
#    The code uses the FFT and is vectorized.
#
#
#                        input parameters:
#
#  fs - m-by-n array of doubles; each row contains n equispaced samples of
#    a function on [0,L) (only the left endpoint f(0) is included, not f(L))
#  m - the number of functions
#  n - the number of samples; assumed to be an even integer (for now)
#  ell - the interval length
#
#
#                        output parameters:
#
#  fzs - m-by-n complex array containing values of the Fourier transform
#    of each vector, defined by the convention
#
#          \what{f}[k] = \int_{0}^{L} f(t) \phi_k(t) dt             (1)
#
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#
#

    nh2 = np.int(n/2)

#
#    frequencies, in DFT ordering
#
    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n

###    wp.prinf('kfs=',kfs,n)

    dt = ell / n
    fzs[0:m,0:n] = dft.fft(fs[0:m,0:n]) * dt



    return()

#
#
#

def polar_grid(rf,nr,na,xs,ys,rfs,angls):

    pp_cos = np.empty([nr,na],dtype='float64')
    pp_sin = np.empty([nr,na],dtype='float64')
#

    npts = nr*na

    dpi = np.pi

    rfs[0:nr+1] = np.linspace(0,rf,nr+1)
    angls[0:na+1] = np.linspace(0,2*dpi,na+1)


    pp_cos[0:nr,0:na] = np.outer(rfs[0:nr],np.cos(angls[0:na]))
    pp_sin[0:nr,0:na] = np.outer(rfs[0:nr],np.sin(angls[0:na]))


    xs[0:npts] = np.reshape(pp_cos[0:nr,0:na],npts)
    ys[0:npts] = np.reshape(pp_sin[0:nr,0:na],npts)


    return()

#
#
#

def ghats_polar(rfs,nr,taus,ngaus,ghats_pol):

    dpi = np.pi
    ghats_pol[0:nr,0:ngaus] = np.exp(-dpi**2 * np.outer(rfs[0:nr]**2,taus[0:ngaus]))


    return()


    for ijk in range(0,ngaus):
        for j in range(0,nr):
            ghats_pol[j,ijk] = np.exp(-dpi**2 * (rfs[j]**2 * taus[ijk]))


    return()

#
#
#



