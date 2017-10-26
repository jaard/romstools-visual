# Module with some visualisation tools for ROMS model output

import numpy as np
from netCDF4 import Dataset as ncread
from scipy.interpolate import griddata
import xarray as xr
import datetime as dt


def spheric_dist(lat1, lat2, lon1, lon2):

    '''
    compute distances for a simple spheric earth

    Parameters
    ----------
    lat1 : float
        latitude of first point (matrix)
    lon1 : float
        longitude of first point (matrix)
    lat2 : float
        latitude of second point (matrix)
    lon2 : float
        longitude of second point (matrix)

    Returns
    -------
    dist : distance from first point to second point (matrix)
    '''

    R = 6367442.76

    # Determine proper longitudinal shift.
    londiff = np.array(abs(lon2-lon1))
    londiff[londiff >= 180] = 360 - londiff[londiff >= 180]

    # Convert Decimal degrees to radians.
    deg2rad = np.pi / 180
    lat1 = lat1 * deg2rad
    lat2 = lat2 * deg2rad
    londiff = londiff * deg2rad

    # Compute the distances

    dist = (R * np.sin(np.sqrt(((np.sin(londiff) * np.cos(lat2)) ** 2) +
            (((np.sin(lat2) * np.cos(lat1)) -
              (np.sin(lat1) * np.cos(lat2) * np.cos(londiff))) ** 2))))

    return dist


def readlat(nc):
    lat = nc.variables['lat_rho'][:]
    if np.size(lat) == 0:
        lat = 1e-5 * nc.variables['y_rho'][:]
        print('READLAT: no horizontal coordinate found')
    return lat


def readlon(nc):
    lon = nc.variables['lon_rho'][:]
    if np.size(lon) == 0:
        lon = 1e-5 * nc['x_rho'][:]
        error('READLON: no horizontal coordinate found')
    return lon


def rho2u_2d(var_rho):

    '''
    interpole a field at rho points to a field at u points

    Parameters
    ----------
    var_rho : ndarray
        variable at rho-points (2D matrix)

    Returns
    -------
     var_u : ndarray
        variable at u-points (2D matrix)
    '''

    Mp, Lp = np.shape(var_rho)
    L = Lp-1
    var_u = 0.5 * (var_rho[:, :L] + var_rho[:, 1:Lp])
    return var_u


def rho2v_2d(var_rho):

    '''
    interpole a field at rho points to a field at v points

    Parameters
    ----------
    var_rho : ndarray
        variable at rho-points (2D matrix)

    Returns
    -------
     var_v : ndarray
        variable at v-points (2D matrix)
    '''

    Mp, Lp = np.shape(var_rho)
    M = Mp-1
    var_v = 0.5 * (var_rho[1:M, :] + var_rho[2:Mp, :])
    return var_v


def read_latlonmask(fname, typ):

    '''
    Read the latitude, the longitude and the mask from a
    netcdf ROMS file

    Parameters
    ----------
    fname : string
        ROMS file name
    typ : string
        type of the variable (character):
        r for 'rho'
        w for 'w'
        u for 'u'
        v for 'v'

    Returns
    -------
    lat : ndarray
        Latitude  (2D matrix)
    lon : ndarray
        Longitude (2D matrix)
    mask : ndarray
        Mask (2D matrix) (1:sea - nan:land)
    '''

    nc = ncread(fname)
    lat = readlat(nc)
    lon = readlon(nc)
    try:
        mask = nc.variables['mask_rho'][:]
        if np.size(mask) == 0:
            mask = 1 + 0 * lon
    except KeyError:
        mask = 1 + 0 * lon
    nc.close()
    Mp, Lp = np.shape(mask)

    if typ == 'u':
        lat = rho2u_2d[lat]
        lon = rho2u_2d[lon]
        mask = mask[:, :Lp-1] * mask[:, 1: Lp]
    elif typ == 'v':
        lat = rho2v_2d[lat]
        lon = rho2v_2d[lon]
        mask = mask[:Mp-1, :] * mask[1: Mp, :]
    mask[mask == 0] = 'nan'

    return lat, lon, mask


def get_type(fname, vname, vlevin):

    '''
    Get the "type" of the ROMS variable rho, u or v point.
    '''

    vlevout = vlevin
    typ = 'r'

    nc = ncread(fname)
    var = nc.variables[vname]
    if var.size == 0:
        typ = ''
        return typ, vlevout

    names = var.dimensions
    ndim = len(names)
    nc.close()
    if ndim == 1:
        typ = ''
        return typ, vlevout

    ind = 0                     # Test if there is a time dimension
    name = names[ind]
    if names[ind] != 'time':
        if names[ind] != 'month':
            print('warning: no time dependency')
        else:
            print('warning: file is a climatology, first dimension is month')
            ind += 1
    else:
        ind += 1

    name = names[ind]
    if name[0] == 's':
        if name[-1] == 'w':
            typ = 'w'
            return typ, vlevout  # dimension is 's_w', return 'w'
        else:
            ind = ind+1
    else:
        vlevout = 0  # If the variable is 2-D, set level to zero

    name = names[ind]
    if name[0] != 'e':
        typ = ''
        return typ, vlevout  # dimension is not 's_...'
    else:                    # or 'eta_...', return ''
        if name[-1] == 'v':
            typ = 'v'
            return typ, vlevout  # dimension is eta_v, return 'v'

    name = names[ind+1]
    if name[0] != 'x':
        typ = ''
        return typ, vlevout  # dimension is not 'xi_...', return ''
    else:
        if name[-1] == 'u':
            typ = 'u'
            return typ, vlevout  # dimension is xi_u, return 'u'

    return typ, vlevout


def oacoef(*args):

    '''
    def oacoef(londata, latdata, lon, lat, ro)
    function extrfield = oacoef(londata,latdata,lon,lat,ro)

    compute an objective analysis on a scalar field.

    Parameters
    ----------
    londata   : longitude of data points (vector)
    latdata   : latitude of data points (vector)
    lon       : longitude of the estimated points (vector)
    lat       : latitude of the estimated points (vector)
    ro        : decorrelation scale

    Returns
    -------
    coef : oa matrix
    extrfield = mdata+coef*(data-mdata)
    '''

    nargin = len(args)
    if nargin < 4:
        raise TypeError('Not enough input arguments')
    elif nargin < 5:
        print('using default decorrelation scale:  ro = 500 km')
        ro = 5e5
    else:
        ro = args[4]

    londata = args[0]
    latdata = args[1]
    lon = args[2]
    lat = args[3]

    i = np.arange(0, len(londata))
    j = np.arange(0, len(lon))
    I, J = np.meshgrid(i, i)
    r1 = spheric_dist(latdata[I], latdata[J], londata[I], londata[J])

    I, J = np.meshgrid(i, j)
    r2 = spheric_dist(lat[J], latdata[I], lon[J], londata[I])

    # np.linalg.lstsq(B, b)
    B = np.array(np.exp(-r2 / ro))
    A = np.array(np.exp(-r1 / ro))
    # coef = B / A
    coef = np.linalg.lstsq(A.T, B.T)[0].T

    return coef


def csf(sc, theta_s, theta_b):

    '''
    function h = csf(sc, theta_s,theta_b)
    '''

    if theta_s > 0:
        csrf = (1-np.cosh(sc * theta_s)) / (np.cosh(theta_s)-1)
    else:
        csrf = -sc ** 2
    if theta_b > 0:
        h = (np.exp(theta_b * csrf)-1) / (1-np.exp(-theta_b))
    else:
        h = csrf

    return h


def zlevs(h, zeta, theta_s, theta_b, hc, N, typ, *args):

    '''
    function z = zlevs(h, zeta, theta_s, theta_b, hc, N, typ, vtransform)

    this function compute the depth of rho or w points for ROMS

    Parameters
    ----------
    typ : string
        'r': rho point 'w': w point
    vtransform : int
        1=> old v transform (Song, 1994)
        2=> new v transform (Shcheptekin, 2006)
    Returns
    -------
    z : ndarray
        Depths (m) of RHO- or W-points (3D matrix).
    '''

    hshape = np.shape(h)
    if len(hshape) == 1:
        L = hshape[0]
        M = 1
    else:
        L, M = np.shape(h)
    if len(args) < 1:
        print(['WARNING no vtransform defined'])
        vtransform = 1  # old vtranform  =  1
        print(['Default S-coordinate system use : Vtransform = 1 (old one)'])
    else:
        vtransform = args[0]
    # Set S-Curves in domain [-1 < sc < 0] at vertical W- and RHO-points.

    sc_r = np.zeros([N, 1])
    Cs_r = np.zeros([N, 1])
    sc_w = np.zeros([N+1, 1])
    Cs_w = np.zeros([N+1, 1])

    if vtransform == 2:
        ds = 1 / N
        if typ == 'w':
            sc_w[0] = -1.0
            sc_w[N] = 0
            Cs_w[0] = -1.0
            Cs_w[N] = 0
            sc_w[1:N] = ds * (np.arange(1, N) - N)
            Cs_w = csf(sc_w, theta_s, theta_b)
            N = N+1
        else:
            sc = ds * (np.arange(1, N+1)-N-0.5)
            Cs_r = csf(sc, theta_s, theta_b)
            sc_r = sc

    else:
        cff1 = 1. / np.sinh(theta_s)
        cff2 = 0.5 / np.tanh(0.5 * theta_s)
        if typ == 'w':
            sc = (np.arange(0, N+1)-N) / N
            N = N+1
        else:
            sc = (np.arange(1, N+1)-N-0.5) / N

        Cs = ((1. - theta_b) * cff1 * np.sinh(theta_s * sc)
              + theta_b * (cff2 * np.tanh(theta_s * (sc+0.5))-0.5))

    #
    # Create S-coordinate system: based on model topography h(i,j),
    # fast-time-averaged free-surface field and vertical coordinate
    # transformation metrics compute evolving depths of of the three-
    # dimensional model grid. Also adjust zeta for dry cells.

    Dcrit = 0.2  # min water depth in dry cells
    h[h == 0] = 1e-14
    zeta[zeta < (Dcrit-h)] = Dcrit - h[zeta < (Dcrit-h)]

    # print('zeta: '+str(zeta))

    hinv = 1. / h
    if M == 1 or L == 1:
        z = np.zeros([N, M, L])
    else:
        z = np.zeros([N, L, M])  # changed because array shape did not match

    if vtransform == 2:
        if typ == 'w':
            cff1 = Cs_w
            cff2 = sc_w+1
            sc = sc_w
        else:
            cff1 = Cs_r
            cff2 = sc_r+1
            sc = sc_r

        h2 = (h+hc)
        cff = hc*sc
        h2inv = 1. / h2

        for k in np.arange(0, N):
            z0 = cff[k] + cff1[k] * h
            z[k, :, :] = z0 * h / (h2) + zeta * (1. + z0 * h2inv)

    else:
        cff1 = Cs
        cff2 = sc + 1
        cff = hc * (sc-Cs)
        cff2 = sc + 1

        for k in np.arange(0, N):
            z0 = cff[k] + cff1[k] * h
            z[k, :, :] = z0 + zeta * (1. + z0 * hinv)
            # print('z0 = '+str(z0))
            # print('z = '+str(z[k, :, :]))

    return z


def tridim(var2d, N):
    '''
    function var3d=tridim(var2d,N)
    '''
    if len(np.shape(var2d)) == 1:
        L = np.shape(var2d)[0]
        M = 1
    else:
        L, M = np.shape(var2d)
    var3d = np.reshape(var2d, (1, M, L))
    var3d = np.tile(var3d, (N, 1, 1))
    return var3d


def get_section(*args):

    ''' def get_section(fname, gname, lonsec, latsec, vname, tindex):

    Extract a vertical slice in any direction (or along a curve)
    from a ROMS netcdf file.

    Parameters
    ----------
    fname : string
        History NetCDF file name.
    gname : string
        Grid NetCDF file name.
    lonsec : list
        Longitudes of the points of the section.
        (vector or [min max] or single value if N-S section).
        (default: [12 18])
    latsec : list
        Latitudes of the points of the section.
        (vector or [min max] or single value if E-W section)
        (default: -34)

    NB: if lonsec and latsec are vectors, they must have the same length.

    vname
        NetCDF variable name to process (character string).
        (default: temp)
    tindex : int
        Netcdf time index (integer).
        (default: 1)

    Returns
    -------
    X           Slice X-distances (km) from the first point (2D matrix).
    Z           Slice Z-positions (matrix).
    VAR         Slice of the variable (matrix).
    '''

    nargin = len(args)
    interp_type = 'linear'

    #
    # Defaults values
    #
    try:
        fname = args[0]
    except IndexError:
        raise NameError('You must specify a file name')
    try:
        gname = args[1]
    except IndexError:
        gname = fname
        print(['Default grid name: ' + gname])
    try:
        lonsec = args[2]
    except IndexError:
        lonsec = [-79, -73]
        print(['Default longitude: ' + str(lonsec)])
    try:
        latsec = args[3]
    except IndexError:
        latsec = -16
        print(['Default latitude: ' + str(latsec)])
    try:
        vname = args[4]
    except IndexError:
        vname = 'temp'
        print(['Default variable to plot: ' + vname])
    try:
        tindex = args[5]
    except IndexError:
        tindex = 1
        print(['Default time index: ' + str(tindex)])

    #
    # Find maximum grid angle size (dl)
    #
    lat, lon, mask = read_latlonmask(gname, 'r')
    M, L = np.shape(lon)
    dl = 1.5 * np.max([np.max(np.abs(lon[1:M, :]-lon[:M-1, :])),
                       np.max(np.abs(lon[:, 1:L]-lon[:, :L-1])),
                       np.max(np.abs(lat[1:M, :]-lat[:M-1, :])),
                       np.max(np.abs(lat[:, 1:L]-lat[:, :L-1]))])
    #
    # Read point positions
    #
    typ, vlevel = get_type(fname, vname, 10)
    if (vlevel == 0):
        print(vname + ' is a 2D-H variable')
        return
    lat, lon, mask = read_latlonmask(gname, typ)
    M, L = np.shape(lon)
    #
    # Find minimal subgrids limits
    #
    minlon = np.min(lonsec)-dl
    minlat = np.min(latsec)-dl
    maxlon = np.max(lonsec)+dl
    maxlat = np.max(latsec)+dl
    sub = (lon > minlon) * (lon < maxlon) * (lat > minlat) * (lat < maxlat)
    if np.sum(sub) == 0:
        print('Section out of the domain')
        return

    ival = np.sum(sub, 0)
    jval = np.sum(sub, 1)
    imin = np.min(np.where(ival != 0))
    imax = np.max(np.where(ival != 0))
    jmin = np.min(np.where(jval != 0))
    jmax = np.max(np.where(jval != 0))
    #
    # Get subgrids
    #
    lon = lon[jmin:jmax+1, imin:imax+1]
    lat = lat[jmin:jmax+1, imin:imax+1]
    sub = sub[jmin:jmax+1, imin:imax+1]
    mask = mask[jmin:jmax+1, imin:imax+1]
    #
    # Put latitudes and longitudes of the section in the correct vector form
    #
    if np.size(lonsec) == 1:
        print('N-S section at longitude: ' + str(lonsec))
        if np.size(latsec) == 1:
            raise ValueError('Need more points to do a section')
        elif len(latsec) == 2:
            latsec = np.arange(latsec[0], latsec[1], dl)

        lonsec = 0 * latsec + lonsec
    elif np.size(latsec) == 1:
        print('E-W section at latitude: ' + str(latsec))
        if len(lonsec) == 2:
            lonsec = np.arange(lonsec[0], lonsec[1], dl)
        latsec = 0 * lonsec + latsec

    elif (len(lonsec) == 2) and (len(latsec) == 2):
        Npts = np.ceil(np.max([np.abs(lonsec[1]-lonsec[0])/dl,
                               np.abs(latsec[1]-latsec[0])/dl]))
        if lonsec[0] == lonsec[1]:
            lonsec = lonsec[0] + np.zeros([1, Npts+1])
        else:
            lonsec = np.arange(lonsec[0], lonsec[1],
                               (lonsec[1]-lonsec[0]) / Npts)
        if latsec[0] == latsec[1]:
            latsec = latsec[0] + np.zeros([1, Npts+1])
        else:
            latsec = np.arange(latsec[0], latsec[1],
                               (latsec[1]-latsec[0]) / Npts)

    elif len(lonsec) != len(latsec):
        raise TypeError('Section latitudes and longitudes are not of' +
                        ' the same length')
    Npts = len(lonsec)
    #
    # Get the subgrid
    #
    sub = 0 * lon
    for ii in np.arange(0, Npts):
        sub[(lon > lonsec[ii]-dl) * (lon < lonsec[ii]+dl) *
            (lat > latsec[ii]-dl) * (lat < latsec[ii]+dl)] = 1
    #
    #  get the coefficients of the objective analysis
    #
    londata = lon[sub == 1]
    latdata = lat[sub == 1]
    coef = oacoef(londata, latdata, lonsec, latsec, 100e3)
    #
    # Get the mask
    #
    mask = mask[sub == 1]
    m1 = griddata((londata, latdata), mask, (lonsec, latsec), method='nearest')
    # mask(isnan(mask)) = 0
    # mask = mean(mask)+coef*(mask-mean(mask))
    # mask(mask>0.5) = 1
    # mask(mask< = 0.5) = NaN
    londata = londata[mask == 1]
    latdata = latdata[mask == 1]
    #
    #  Get the vertical levels
    #
    nc = ncread(gname)
    h = nc.variables['h'][:]
    hmin = np.min(h)
    h = h[jmin:jmax+1, imin:imax+1]
    nc.close()
    h = h[sub == 1]
    h = h[mask == 1]
    # h = mean(h)+coef*(h-mean(h))
    h = griddata((londata, latdata), h, (lonsec, latsec), method=interp_type)
    #
    nc = ncread(fname)

    # zeta = np.squeeze(nc.variables['zeta'][tindex, jmin:jmax+1, imin:imax+1])
    try:
        zeta = np.squeeze(nc.variables['zeta'][tindex, jmin:jmax+1,
                                               imin:imax+1])
        if np.size(zeta) == 0:
            zeta = 0 * h
        else:
            zeta = zeta[sub == 1]
            zeta = zeta[mask == 1]
            # zeta = mean(zeta)+coef*(zeta-mean(zeta))
            zeta = griddata((londata, latdata), zeta, (lonsec, latsec),
                            method=interp_type)
    except KeyError:
        zeta = 0 * h

    theta_s = nc.theta_s

    if np.size(theta_s) == 0:
        # print('Rutgers version')
        theta_s = nc.variables['theta_s']
        theta_b = nc.variables['theta_b']
        Tcline = nc.variables['Tcline']
    else:
        # print('UCLA version')
        theta_b = nc.theta_b
        Tcline = nc.Tcline

    if np.size(Tcline) == 0:
        # print('UCLA version 2')
        hc = nc.hc
    else:
        hmin = np.nanmin(h)
        hc = np.min([hmin, Tcline])

    N = len(nc.variables['s_rho'][:])
    s_coord = 1
    try:
        VertCoordType = nc.VertCoordType
        if VertCoordType == 'NEW':
            s_coord = 2
    except AttributeError:
        try:
            vtrans = nc.variables['Vtransform'][:]
            if np.size(vtrans) != 0:
                s_coord = vtrans
        except KeyError:
            pass
    if s_coord == 2:
        hc = Tcline

    # print('h: '+str(h))
    # print('hc: '+str(hc))
    # print('theta_s: '+str(theta_s))
    # print('theta_b: '+str(theta_b))
    Z = np.squeeze(zlevs(h, zeta, theta_s, theta_b, hc, N, typ, s_coord))
    N, Nsec = np.shape(Z)
    #
    # Loop on the vertical levels
    #
    VAR = 0 * Z
    for k in np.arange(0, N):
        var = np.squeeze(nc.variables[vname][tindex, k, jmin:jmax+1,
                                             imin:imax+1])
        var = var[sub == 1]
        var = var[mask == 1]
        var = griddata((londata, latdata), var,
                       (lonsec, latsec), method=interp_type)
        VAR[k, :] = m1 * var

    nc.close()
    #
    # Get the distances
    #
    dist = spheric_dist(latsec[0], latsec, lonsec[0], lonsec)/1e3
    X = np.squeeze(tridim(dist, N))
    # X[np.isnan(Z)] = 'nan'

    return X, Z, VAR


def vinterp(var, z, depth):

    '''
    function  vnew = vinterp(var,z,depth)

    This function interpolates a 3D variable on a horizontal level of constant
    depth

    Parameters
    ----------
    var     Variable to process (3D matrix).
    z       Depths (m) of RHO- or W-points (3D matrix).
    depth   Slice depth (scalar; meters, negative).

    Returns
    -------
    vnew    Horizontal slice (2D matrix).
    '''

    N, Mp, Lp = np.shape(z)
    #
    # Find the grid position of the nearest vertical levels
    #
    a = z < depth
    levs = np.squeeze(np.sum(a, 0))
    levs[levs == N] = N-1
    mask = np.ones(levs.shape, dtype=bool)
    mask[levs == 0] = False

    imat, jmat = np.meshgrid(np.arange(0, Lp), np.arange(0, Mp))
    pos = N * Mp * (imat) + N * (jmat) + levs
    pos[np.invert(mask)] = 1
    #
    # Do the interpolation
    #
    z1 = z.flatten('F')[pos]
    z2 = z.flatten('F')[pos-1]
    v1 = var.flatten('F')[pos]
    v2 = var.flatten('F')[pos-1]
    vnew = mask * (((v1-v2) * depth + v2 * z1-v1 * z2) / (z1-z2))
    vnew[np.invert(mask)] = 'nan'

    return vnew


def rho2u_3d(var_rho):

    '''
    function var_u=rho2u_3d(var_rho)

    interpole a field at rho points to a field at u points

    Parameters
    ----------
    var_rho variable at rho-points (3D matrix)

    Returns
    -------
    var_u   variable at u-points (3D matrix)
    '''

    N, Mp, Lp = np.shape(var_rho)
    L = Lp - 1
    var_u = 0.5 * (var_rho[:, :, 0:L] + var_rho[:, :, 1:Lp])

    return var_u


def rho2v_3d(var_rho):

    '''
    function var_v = rho2v_3d(var_rho)

    interpole a field at rho points to a field at v points

    Parameters
    ----------
    var_rho variable at rho-points (3D matrix)

    Returns
    -------
    var_v   variable at v-points (3D matrix)
    '''

    N, Mp, Lp = np.shape(var_rho)
    M = Mp - 1
    var_v = 0.5 * (var_rho[:, 0:M, :] + var_rho[:, 1:Mp, :])

    return var_v


def u2rho_3d(var_u):

    N, Mp, L = np.shape(var_u)
    Lp = L + 1
    Lm = L - 1
    var_rho = np.zeros((N, Mp, Lp))
    var_rho[:, :, 1:L] = 0.5 * (var_u[:, :, :Lm] + var_u[:, :, 1:L])
    var_rho[:, :, 0] = var_rho[:, :, 1]
    var_rho[:, :, L] = var_rho[:, :, Lm]

    return var_rho


def v2rho_3d(var_v):

    N, M, Lp = np.shape(var_v)
    Mp = M + 1
    Mm = M - 1
    var_rho = np.zeros((N, Mp, Lp))
    var_rho[:, 1:M, :] = 0.5 * (var_v[:, :Mm, :] + var_v[:, 1:M, :])
    var_rho[:, 0, :] = var_rho[:, 1, :]
    var_rho[:, M, :] = var_rho[:, Mm, :]

    return var_rho


def u2rho_3d_xr(var_u, rhoarray):

    TS, N, Mp, L = np.shape(var_u)
    Lp = L + 1
    Lm = L - 1
    var_rho = rhoarray * 0
    var_rho = var_rho.rename(var_u.name)
    var_rho[:, :, :, 1:L] = 0.5 * (var_u[:, :, :, :Lm] + var_u[:, :, :, 1:L])
    # var_rho[:, :, :, 0] = var_rho[:, :, :, 1]
    # var_rho[:, :, :, L] = var_rho[:, :, :, Lm]

    return var_rho


def v2rho_3d_xr(var_v, rhoarray):

    TS, N, M, Lp = np.shape(var_v)
    Mp = M + 1
    Mm = M - 1
    var_rho = rhoarray * 0
    var_rho = var_rho.rename(var_v.name)
    var_rho[:, :, 1:M, :] = 0.5 * (var_v[:, :, :Mm, :] + var_v[:, :, 1:M, :])
    # var_rho[:, :, 0, :] = var_rho[:, :, 1, :]
    # var_rho[:, :, M, :] = var_rho[:, :, Mm, :]

    return var_rho


def get_depths(fname, gname, tindex, typ):

    '''
    Get the depths of the sigma levels
    '''
    nc = ncread(gname)
    h = nc.variables['h'][:]
    nc.close()

    #
    # Open history file
    #
    nc = ncread(fname)
    zeta = np.squeeze(nc.variables['zeta'][tindex, :, :])
    theta_s = nc.theta_s
    if np.size(theta_s) == 0:
        # print('Rutgers version')
        theta_s = nc.variables['theta_s'][:]
        theta_b = nc.variables['theta_b'][:]
        Tcline = nc.variables['Tcline'][:]
    else:
        # print('AGRIF/UCLA version')
        theta_b = nc.theta_b
        Tcline = nc.Tcline
        hc = nc.hc

    if np.size(Tcline) == 0:
        # print('UCLA version')
        hc = nc.hc
    else:
        hmin = np.nanmin(h)
        hc = np.min([hmin, Tcline])

    N = len(nc.variables['s_rho'][:])
    s_coord = 1
    try:
        VertCoordType = nc.VertCoordType
        if VertCoordType == 'NEW':
            s_coord = 2
    except AttributeError:
        try:
            vtrans = nc.variables['Vtransform'][:]
            if np.size(vtrans) != 0:
                s_coord = vtrans
        except KeyError:
            pass
    if s_coord == 2:
        hc = Tcline
    nc.close()

    if np.size(zeta) == 0:
        zeta = 0. * h

    vtyp = typ
    if (typ == 'u') or (typ == 'v'):
        vtyp = 'r'

    z = zlevs(h, zeta, theta_s, theta_b, hc, N, vtyp, s_coord)
    if typ == 'u':
        z = rho2u_3d(z)
    if typ == 'v':
        z = rho2v_3d(z)

    return z


def get_hslice(fname, gname, vname, tindex, level, typ, *args):

    '''
    function var=get_hslice(fname,gname,vname,tindex,level,typ)
    get an horizontal slice of a ROMS variable
    Parameters
    ----------
    fname    ROMS netcdf file name (average or history) (string)
    gname    ROMS netcdf grid file name  (string)
    vname    name of the variable (string)
    tindex   time index (integer)
    level    vertical level of the slice (scalar):
         level =   integer >= 1 and <= N
                   take a slice along a s level (N=top))
         level =   0
                   2D horizontal variable (like zeta)
         level =   real < 0
                   interpole a horizontal slice at z=level
    typ    type of the variable (character):
         r for 'rho' for zeta, temp, salt, w(!)
         w for 'w'   for AKt
         u for 'u'   for u, ubar
         v for 'v'   for v, vbar
    Returns
    -------
    var     horizontal slice (2D matrix)
    '''

    nc = ncread(fname)
    if level == 0:
        #
        # 2D variable
        #
        var = np.squeeze(nc.variables[vname][tindex, :, :])
    elif level > 0:
        #
        # Get a sigma level of a 3D variable
        #
        var = np.squeeze(nc.variables[vname][tindex, level, :, :])
    else:
        #
        # Get a horizontal level of a 3D variable
        #
        # Get the depths of the sigma levels
        #
        # if len(args) != 0:
        if args:
            z = get_depths(args[0], gname, tindex, typ)
        else:
            z = get_depths(fname, gname, tindex, typ)
        #
        # Read the 3d matrix and do the interpolation
        #
        var_sigma = np.squeeze(nc.variables[vname][tindex, :, :, :])
        var = vinterp(var_sigma, z, level)

    nc.close()
    g = ncread(gname)
    if typ == 'r' or typ == 'w':
        mask = g.variables['mask_rho'][:]
    elif typ == 'u':
        mask = g.variables['mask_u'][:]
    elif typ == 'v':
        mask = g.variables['mask_v'][:]
    # apply the landmask
    # mask = g.variables['mask_rho'][:].data
    if type(mask) == np.ma.core.MaskedArray:
        mask = mask.data
    mask[mask == 0] = np.nan
    var *= mask
    var = np.array(var)

    return var


def gauss(x, *args):

    '''
    function y = gauss(x, s, m)

    Computes the Gaussian function of the input array

    y=exp(-(x-m).^2./s.^2)./(sqrt(2*pi).*s);
    Bronstein p. 81
    '''

    try:
        m = args[1]
    except IndexError:
        m = 0
    try:
        s = args[0]
    except IndexError:
        s = 1

    y = np.exp(-(x - m)**2 / s**2) / (np.sqrt(2 * np.pi) * s)

    return y


def get_chla(fname):

    '''
    Calculates Chlorophyll A in a ROMS file 'fname' from SPHYTO and LPHYTO
    and returns an xarray.Dataarray CHLA
    '''

    theta_m = 0.020
    CN_Phyt = 6.625
    rf = xr.open_dataset(fname)

    chla = theta_m * (rf.SPHYTO + rf.LPHYTO) * CN_Phyt * 12
    # chla[chla <= 0] = np.nan

    # rf['CHLA'] = chla

    return chla, rf


# This returns the wrong date for the runs we have!
def get_date(fname, tindex, *args):

    '''
    get the date from the time index of a ROMS netcdf file
    (for a 360 days year)

    if Yorig (year origin) is provided, the date is computed
    as a "real time" date.

    ATTENTION:
    Time index starts with "0"!

    Parameters
    ----------
    fname    ROMS netcdf file name (average or history) (string)
    tindex   time index (integer) <--- first step is "0"
                                       ("1" in the MATLAB version)

    Returns
    -------
    day      day (scalar)
    month    month (string)
    year     year (scalar)
    thedate  date (string)
    '''

    if args:
        Yorig = args[0]
    else:
        Yorig = np.nan

    nc = ncread(fname)
    time = nc.variables['scrum_time'][tindex]
    if np.size(time) == 0:
        time = nc.variables['ocean_time'][tindex]
    if np.size(time) == 0:
        year = 0
        imonth = 0
        day = 0
        month = ''
        thedate = ''
        nc.close()
        return day, month, year, imonth, thedate
    nc.close()

    Month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
             'Sep', 'Oct', 'Nov', 'Dec']

    if np.isnan(Yorig):
        # Climatology simulation
        year = floor(1+time/(24*3600*360))
        if time <= 0:
            time = time-(year-1)*24*3600*360

        imonth = floor(1+np.mod(time/(24*3600*30), 12))
        day = (1+np.mod(time/(24*3600), 30))
        # day = floor(1+np.mod(time/(24*3600), 30)) <--- original
    else:
        # "Real time" simulation
        datetime_obj = dt.timedelta(time/(24*3600)) + dt.datetime(Yorig, 1, 1)
        year = datetime_obj.year
        imonth = datetime_obj.month
        day = datetime_obj.day
        month = Month[imonth-1]
        # thedate = str(datetime_obj.date())
        thedate = str(day) + ' ' + month + ' ' + str(year)

    return thedate, datetime_obj()


def psi2rho(var_psi):
    '''
    Transfer a field at psi points to the rho points
    '''

    M, L = np.shape(var_psi)
    Mp = M + 1
    Lp = L + 1
    Mm = M - 1
    Lm = L - 1
    var_rho = np.zeros((Mp, Lp))
    var_rho[1:M, 1:L] = (0.25 * (var_psi[:Mm, :Lm] + var_psi[:Mm, 1:L] +
                         var_psi[1:M, :Lm] + var_psi[1:M, 1:L]))
    var_rho[0, :] = var_rho[1, :]
    var_rho[M, :] = var_rho[Mm, :]
    var_rho[:, 0] = var_rho[:, 1]
    var_rho[:, L] = var_rho[:, Lm]

    return var_rho


def vorticity(ubar, vbar, pm, pn):
    '''
    Computes the relative vorticity
    '''

    Mp, Lp = np.shape(pm)
    # L = Lp
    # M = Mp
    L = Lp-1
    M = Mp-1
    xi = np.zeros((M, L))
    mn_p = np.zeros((M, L))
    uom = np.zeros((M, Lp))
    von = np.zeros((Mp, L))
    uom = 2 * ubar / (pm[:, :L] + pm[:, 1:Lp])
    von = 2 * vbar / (pn[0:M, :] + pn[1:Mp, :])
    mn = pm * pn
    mn_p = ((mn[0:M, 0:L] + mn[0:M, 1:Lp] +
             mn[1:Mp, 1:Lp] + mn[1:Mp, 0:L])/4)
    xi = mn_p * (von[:, 1:Lp] - von[:, 0:L] - uom[1:Mp, :] + uom[0:M, :])

    return xi


def get_vort(fname, gname, tindex, vlevel):
    '''
    Compute the vorticity at rho points
    '''

    #
    # Get the grid parameters
    #
    lat, lon, mask = read_latlonmask(gname, 'r')
    mask[2:-1, 2:-1] = (mask[1:-2, 1:-2] *
                        mask[1:-2, 2:-1] *
                        mask[1:-2, 3:] *
                        mask[2:-1, 1:-2] *
                        mask[2:-1, 2:-1] *
                        mask[2:-1, 3:] *
                        mask[3:, 1:-2] *
                        mask[3:, 2:-1] *
                        mask[3:, 3:])
    nc = ncread(gname)
    pm = np.array(nc.variables['pm'][:])
    pn = np.array(nc.variables['pn'][:])
    nc.close()
    #
    # Get the currents
    #
    if vlevel == 0:
        u = get_hslice(fname, gname, 'ubar', tindex, vlevel, 'u')
        v = get_hslice(fname, gname, 'vbar', tindex, vlevel, 'v')
    else:
        u = get_hslice(fname, gname, 'u', tindex, vlevel, 'u')
        v = get_hslice(fname, gname, 'v', tindex, vlevel, 'v')
    #
    # Get vorticity at rho points
    #
    # xi = coef * mask * psi2rho(vorticity(u, v, pm, pn))
    xi = mask * psi2rho(vorticity(u, v, pm, pn))

    if type(xi) == np.ma.core.MaskedArray:
        xi = xi.data
    # return lat.data, lon.data, mask.data, xi.data
    return xi




# function [dvdz,zw]=vert_grad(var,zr)
# %
# %
# %  Get the vertical gradient a 3D ROMS variable
# %
# %  Input:
# %         var  ROMS 3D matrix
# %         zr   ROMS 3D matrix of the z levels
# %  Output:
# %         dvdz Vertical gradient
# %         zw   z levels
# %
# %  Further Information:
# %  http://www.brest.ird.fr/Roms_tools/
# %
# %  This file is part of ROMSTOOLS
# %
# %  ROMSTOOLS is free software; you can redistribute it and/or modify
# %  it under the terms of the GNU General Public License as published
# %  by the Free Software Foundation; either version 2 of the License,
# %  or (at your option) any later version.
# %
# %  ROMSTOOLS is distributed in the hope that it will be useful, but
# %  WITHOUT ANY WARRANTY; without even the implied warranty of
# %  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# %  GNU General Public License for more details.
# %
# %  You should have received a copy of the GNU General Public License
# %  along with this program; if not, write to the Free Software
# %  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# %  MA  02111-1307  USA
# %
# %  Copyright (c) 2006 by Pierrick Penven
# %  e-mail:Pierrick.Penven@ird.fr
# %
# %
#
#
#   zw=0.5*(zr(2:end,:,:)+zr(1:end-1,:,:));
#   dvdz=(var(2:end,:,:)-var(1:end-1,:,:))./...
#          (zr(2:end,:,:)-zr(1:end-1,:,:));
# return



def u2rho_2d(var_u):

    '''
    transfer a field at u points to a field at rho points

    Parameters
    ----------
    var_u : ndarray
        variable at u-points (2D matrix)

    Returns
    -------
    var_rho : ndarray
        variable at rho-points (2D matrix)
    '''

    Mp, L = np.shape(var_u)
    Lp = L + 1
    Lm = L - 1
    var_rho = np.zeros([Mp, Lp])
    var_rho[:, 1:L] = 0.5 * (var_u[:, :Lm] + var_u[:, 1:L])
    var_rho[:, 0] = var_rho[:, 1]
    var_rho[:, L] = var_rho[:, Lm]

    return var_rho



def v2rho_2d(var_v):

    '''
    transfer a field at v points to a field at rho points

    Parameters
    ----------
    var_v : ndarray
        variable at v-points (2D matrix)

    Returns
    -------
    var_rho : ndarray
        variable at rho-points (2D matrix)
    '''

    M, Lp = np.shape(var_v)
    Mp = M + 1
    Mm = M - 1
    var_rho = np.zeros([Mp, Lp])
    var_rho[1:M, :] = 0.5 * (var_v[:Mm, :]+var_v[1:M, :])
    var_rho[0, :] = var_rho[1, :]
    var_rho[M, :] = var_rho[Mm, :]

    return var_rho


def u2rho_2d_time(var_u):

    '''
    transfer a field at u points to a field at rho points

    Parameters
    ----------
    var_u : ndarray
        variable at u-points (2D matrix)

    Returns
    -------
    var_rho : ndarray
        variable at rho-points (2D matrix)
    '''

    T, Mp, L = np.shape(var_u)
    Lp = L + 1
    Lm = L - 1
    var_rho = np.zeros([T, Mp, Lp])
    var_rho[:, :, 1:L] = 0.5 * (var_u[:, :, :Lm] + var_u[:, :, 1:L])
    var_rho[:, :, 0] = var_rho[:, :, 1]
    var_rho[:, :, L] = var_rho[:, :, Lm]

    return var_rho



def v2rho_2d_time(var_v):

    '''
    transfer a field at v points to a field at rho points

    Parameters
    ----------
    var_v : ndarray
        variable at v-points (2D matrix)

    Returns
    -------
    var_rho : ndarray
        variable at rho-points (2D matrix)
    '''

    T, M, Lp = np.shape(var_v)
    Mp = M + 1
    Mm = M - 1
    var_rho = np.zeros([T, Mp, Lp])
    var_rho[:, 1:M, :] = 0.5 * (var_v[:, :Mm, :]+var_v[:, 1:M, :])
    var_rho[:, 0, :] = var_rho[:, 1, :]
    var_rho[:, M, :] = var_rho[:, Mm, :]

    return var_rho



def get_ke(fname, gname, tindex, vlevel, **kwargs):

    '''
    Compute the kinetic Energy
    '''

    lat, lon, mask = read_latlonmask(gname,'r')
    if vlevel == 0:
        #            get_hslice(fname, gname, 'ubar', tindex, vlevel, 'u')
        u = u2rho_2d(get_hslice(fname, gname, 'ubar', tindex, vlevel, 'u'))
        v = v2rho_2d(get_hslice(fname, gname, 'vbar', tindex, vlevel, 'v'))
    else:
        try:
            u = u2rho_2d(get_hslice(fname, gname, 'u', tindex, vlevel, 'u'))
            v = v2rho_2d(get_hslice(fname, gname, 'v', tindex, vlevel, 'v'))
        except KeyError:
            u = u2rho_2d(get_hslice(fname, gname, kwargs['u'], tindex, vlevel, 'u'))
            v = v2rho_2d(get_hslice(fname, gname, kwargs['v'], tindex, vlevel, 'v'))

    ke = mask * 0.5 * (u**2 + v**2)

    if type(ke) == np.ma.core.MaskedArray:
        ke = ke.data

    return lat, lon, mask, ke



def get_ke_2d(dset, gname, tindex, **kwargs):

    '''
    Compute the kinetic Energy from 2d field
    '''

    lat, lon, mask = read_latlonmask(gname,'r')
    if kwargs:
        u = u2rho_2d(dset[kwargs['u']][tindex, :, :])
        v = v2rho_2d(dset[kwargs['v']][tindex, :, :])
    else:
        u = u2rho_2d(dset.u[:, :, :].data)
        v = v2rho_2d(dset.v[:, :, :].data)


    ke = mask * 0.5 * (u**2 + v**2)

    if type(ke) == np.ma.core.MaskedArray:
        ke = ke.data

    return lat, lon, mask, ke


def get_ke_2d_time(dset, gname, **kwargs):

    '''
    Compute the kinetic Energy from 2d field
    '''

    lat, lon, mask = read_latlonmask(gname,'r')
    if kwargs:
        u = u2rho_2d_time(dset[kwargs['u']][:, :, :].data)
        v = v2rho_2d_time(dset[kwargs['v']][:, :, :].data)
    else:
        u = u2rho_2d_time(dset.u[:, :, :].data)
        v = v2rho_2d_time(dset.v[:, :, :].data)


    ke = mask * 0.5 * (u**2 + v**2)

    if type(ke) == np.ma.core.MaskedArray:
        ke = ke.data

    return lat, lon, mask, ke



# def rho2u_2d(var_rho):
#
#     '''
#     interpole a field at rho points to a field at u points
#
#     Parameters
#     ----------
#     var_rho : ndarray
#         variable at rho-points (2D matrix)
#
#     Returns
#     -------
#      var_u : ndarray
#         variable at u-points (2D matrix)
#     '''
#
#     Mp, Lp = np.shape(var_rho)
#     L = Lp-1
#     var_u = 0.5 * (var_rho[:, :L] + var_rho[:, 1:Lp])
#     return var_u
