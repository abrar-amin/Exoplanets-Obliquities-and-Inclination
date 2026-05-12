import numpy as np
import jax.numpy as jnp
import jax
import sys
from jaxoplanet.orbits.keplerian import Central
from jaxoplanet.starry.orbit import SurfaceSystem, SurfaceBody
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.starry.light_curves import light_curve
import scipy.constants as sc
import scipy.interpolate as spi
from numba import jit, literal_unroll

import pca
import dummyfit
import matplotlib.pyplot as plt
from jaxoplanet.starry.visualization import show_surface


def initsystem(fit, ydeg, y=None):
    '''
    Uses a fit object to build the respective jaxoplanet objects. Useful
    because jaxoplanet objects cannot be pickled. Returns a tuple of
    (star_surface, planet_surface, system).

    Arguments
    ---------
    fit: Fit object
        Fit object with configuration loaded.

    ydeg: int
        Maximum spherical harmonic degree for the planet map.

    y: 1D array, optional
        Ylm coefficients for the planet map. If None, a uniform map
        (Y00=1, all others 0) is used. Y00 is always forced to 1.
    '''
    cfg = fit.cfg
    star_ylm = Ylm.from_dense(jnp.array([1.0]), normalize=False)

    star_surface = Surface(
        y=star_ylm,
        inc=jnp.pi/2,
        period=cfg.star.prot,
        radius=cfg.star.r,
        u=(),
        normalize=False,
        amplitude=1.0,
    )

    if y is None:
        n_coeffs = (ydeg + 1)**2
        planet_ylm_coeffs = jnp.zeros(n_coeffs)
        planet_ylm_coeffs = planet_ylm_coeffs.at[0].set(1.0)
    else:
        planet_ylm_coeffs = jnp.array(y)
        planet_ylm_coeffs = planet_ylm_coeffs.at[0].set(1.0)

    planet_ylm = Ylm.from_dense(planet_ylm_coeffs, normalize=False)

    planet_surface = Surface(
        y=planet_ylm,
        inc=jnp.deg2rad(cfg.planet.inc),
        period=cfg.planet.prot,
        radius=cfg.planet.r,
        u=(),
        normalize=False,
        amplitude=1.0,
        phase=jnp.deg2rad(180),
    )

    central = Central(
        mass=cfg.star.m,
        radius=cfg.star.r,
    )

    system = SurfaceSystem(
        central=central,
        central_surface=star_surface,
    )

    system = system.add_body(
        period=cfg.planet.porb,
        radius=cfg.planet.r,
        mass=cfg.planet.m,
        inclination=jnp.deg2rad(cfg.planet.inc),
        eccentricity=cfg.planet.ecc,
        omega_peri=jnp.deg2rad(cfg.planet.w),
        asc_node=jnp.deg2rad(cfg.planet.Omega),
        time_transit=cfg.planet.t0,
        surface=planet_surface,
    )

    return star_surface, planet_surface, system


def vislon(system, data):
    """
    Determines the range of visible longitudes based on times of observation.

    Arguments
    ---------
    system: jaxoplanet SurfaceSystem
        System object with at least one orbiting body.

    data: Dataset object
        Must contain observation times in data.t.

    Returns
    -------
    minlon: float
        Minimum visible longitude, in degrees.

    maxlon: float
        Maximum visible longitude, in degrees.
    """
    t = data.t

    psurf = system.body_surfaces[0]
    pbody = system.bodies[0]

    prot   = psurf.period
    t0     = pbody.time_transit
    theta0 = psurf.phase * 180.0 / np.pi

    centlon = theta0 - (t - t0) / prot * 360
    limb1 = centlon - 90
    limb2 = centlon + 90
    limb1 = (limb1 + 180) % 360 - 180
    limb2 = (limb2 + 180) % 360 - 180

    return np.min(limb1), np.max(limb2)


def eval_obliquities(system, t, lmax, y00, n_obl=10):
    """
    Compute and plot planet flux light curves for each spherical harmonic (l, m)
    across a range of obliquity values from 0 to pi.

    One subplot is produced per obliquity value. Within each subplot, each
    spherical harmonic Y(l,m) is shown as a separate labeled line, with the
    uniform-map contribution (y00) subtracted so that only the harmonic's
    contribution to the light curve is shown.

    Arguments
    ---------
    system: SurfaceSystem
        A jaxoplanet SurfaceSystem with a star and planet.

    t: 1D array
        Times at which to evaluate the light curve.

    lmax: int
        Maximum spherical harmonic degree.

    y00: 1D array
        Light curve of a normalized, uniform map (same meaning as in mkcurves).

    n_obl: int
        Number of obliquity values to sample between 0 and pi (inclusive).
    """
    planet_surface = system.body_surfaces[0]
    planet_body = system.bodies[0]
    central = system.central
    central_surface = system.central_surface

    obliquities = np.linspace(0, np.pi, n_obl)

    n_coeffs = (lmax + 1)**2
    nharm = n_coeffs - 1

    harm_labels = []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            harm_labels.append(f"Y({l},{m:+d})")

    def make_surface(yval, obl):
        ylm_coeffs = jnp.zeros(n_coeffs)
        ylm_coeffs = ylm_coeffs.at[0].set(1.0)
        ylm_coeffs = ylm_coeffs.at[1:].set(yval)
        new_ylm = Ylm.from_dense(ylm_coeffs, normalize=False)
        return Surface(
            y=new_ylm,
            inc=planet_surface.inc,
            period=planet_surface.period,
            radius=planet_surface.radius,
            u=(),
            normalize=False,
            amplitude=1.0,
            phase=planet_surface.phase,
            obl=float(obl),
        )

    def evalflux_harm(surf):
        new_system = SurfaceSystem(central=central, central_surface=central_surface)
        new_system = new_system.add_body(
            period=planet_body.period,
            radius=planet_body.radius,
            mass=planet_body.mass,
            inclination=planet_body.inclination,
            eccentricity=planet_body.eccentricity,
            omega_peri=planet_body.omega_peri,
            time_transit=planet_body.time_transit,
            surface=surf,
        )
        flux_result = light_curve(new_system, order=20)(t)
        planetflux = np.array(flux_result.T[1])
        return planetflux - y00

    all_surfaces = [[None] * nharm for _ in range(n_obl)]
    all_lcs = np.zeros((n_obl, nharm, len(t)))
    for oi, obl in enumerate(obliquities):
        for hi in range(nharm):
            yval = np.zeros(nharm)
            yval[hi] = 1.0
            surf = make_surface(yval, obl)
            all_surfaces[oi][hi] = surf
            all_lcs[oi, hi] = evalflux_harm(surf)

    inc_deg = np.rad2deg(float(planet_body.inclination))

    lc_ncols = min(n_obl, 5)
    lc_nrows = (n_obl + lc_ncols - 1) // lc_ncols
    fig1, axes1 = plt.subplots(lc_nrows, lc_ncols,
                               figsize=(4 * lc_ncols, 3 * lc_nrows), squeeze=False)
    fig1.suptitle(f"Harmonic Light Curves vs Obliquity  |  inc={inc_deg:.1f}°")
    for oi, obl in enumerate(obliquities):
        ax = axes1[oi // lc_ncols][oi % lc_ncols]
        obl_deg = np.rad2deg(float(obl))
        for hi in range(nharm):
            ax.plot(t, all_lcs[oi, hi], label=harm_labels[hi])
        ax.set_title(f"obl={obl_deg:.1f}°")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("ΔFlux")
        ax.legend(loc='best', fontsize='x-small')
    for oi in range(n_obl, lc_nrows * lc_ncols):
        axes1[oi // lc_ncols][oi % lc_ncols].set_visible(False)
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(nharm, n_obl,
                               figsize=(2 * n_obl, 2.5 * nharm), squeeze=False)
    fig2.suptitle(f"Surface Maps  |  inc={inc_deg:.1f}°")
    for hi in range(nharm):
        for oi, obl in enumerate(obliquities):
            ax = axes2[hi][oi]
            show_surface(all_surfaces[oi][hi], ax=ax, theta=0)
            if hi == 0:
                obl_deg = np.rad2deg(float(obl))
                ax.set_title(f"obl={obl_deg:.1f}°", fontsize=7)
            if oi == 0:
                ax.set_ylabel(harm_labels[hi], fontsize=7)
    fig2.tight_layout()

    plt.show()


def eval_inclination(system, t, lmax, y00, n_inclinations=10):
    """
    Compute and plot planet flux light curves for each spherical harmonic (l, m)
    across a range of inclination values from pi/2 to 3*pi/2.

    One subplot is produced per inclination value. Within each subplot, each
    spherical harmonic Y(l,m) is shown as a separate labeled line, with the
    uniform-map contribution (y00) subtracted so that only the harmonic's
    contribution to the light curve is shown.

    Arguments
    ---------
    system: SurfaceSystem
        A jaxoplanet SurfaceSystem with a star and planet.

    t: 1D array
        Times at which to evaluate the light curve.

    lmax: int
        Maximum spherical harmonic degree.

    y00: 1D array
        Light curve of a normalized, uniform map (same meaning as in mkcurves).

    n_inclinations: int
        Number of inclination values to sample between pi/2 and 3*pi/2 (inclusive).
    """
    planet_surface = system.body_surfaces[0]
    planet_body = system.bodies[0]
    central = system.central
    central_surface = system.central_surface

    inclinations = np.linspace(np.pi / 2, 3 * np.pi / 2, n_inclinations)

    n_coeffs = (lmax + 1)**2
    nharm = n_coeffs - 1

    harm_labels = []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            harm_labels.append(f"Y({l},{m:+d})")

    def make_surface(yval, incline):
        ylm_coeffs = jnp.zeros(n_coeffs)
        ylm_coeffs = ylm_coeffs.at[0].set(1.0)
        ylm_coeffs = ylm_coeffs.at[1:].set(yval)
        new_ylm = Ylm.from_dense(ylm_coeffs, normalize=False)
        return Surface(
            y=new_ylm,
            inc=incline,
            period=planet_surface.period,
            radius=planet_surface.radius,
            u=(),
            normalize=False,
            amplitude=1.0,
            phase=planet_surface.phase,
            obl=planet_surface.obl,
        )

    def evalflux_harm(surf):
        new_system = SurfaceSystem(central=central, central_surface=central_surface)
        new_system = new_system.add_body(
            period=planet_body.period,
            radius=planet_body.radius,
            mass=planet_body.mass,
            inclination=planet_body.inclination,
            eccentricity=planet_body.eccentricity,
            omega_peri=planet_body.omega_peri,
            time_transit=planet_body.time_transit,
            surface=surf,
        )
        flux_result = light_curve(new_system, order=20)(t)
        planetflux = np.array(flux_result.T[1])
        return planetflux - y00

    all_surfaces = [[None] * nharm for _ in range(n_inclinations)]
    all_lcs = np.zeros((n_inclinations, nharm, len(t)))
    for ii, inc in enumerate(inclinations):
        for hi in range(nharm):
            yval = np.zeros(nharm)
            yval[hi] = 1.0
            surf = make_surface(yval, inc)
            all_surfaces[ii][hi] = surf
            all_lcs[ii, hi] = evalflux_harm(surf)

    obl_deg = np.rad2deg(float(planet_surface.obl))

    lc_ncols = min(n_inclinations, 5)
    lc_nrows = (n_inclinations + lc_ncols - 1) // lc_ncols
    fig1, axes1 = plt.subplots(lc_nrows, lc_ncols,
                               figsize=(4 * lc_ncols, 3 * lc_nrows), squeeze=False)
    fig1.suptitle(f"Harmonic Light Curves vs Inclination  |  obl={obl_deg:.1f}°")
    for ii, inc in enumerate(inclinations):
        ax = axes1[ii // lc_ncols][ii % lc_ncols]
        inc_deg = np.rad2deg(float(inc))
        for hi in range(nharm):
            ax.plot(t, all_lcs[ii, hi], label=harm_labels[hi])
        ax.set_title(f"inc={inc_deg:.1f}°")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("ΔFlux")
        ax.legend(loc='best', fontsize='x-small')
    for ii in range(n_inclinations, lc_nrows * lc_ncols):
        axes1[ii // lc_ncols][ii % lc_ncols].set_visible(False)
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(nharm, n_inclinations,
                               figsize=(2 * n_inclinations, 2.5 * nharm), squeeze=False)
    fig2.suptitle(f"Surface Maps  |  obl={obl_deg:.1f}°")
    for hi in range(nharm):
        for ii, inc in enumerate(inclinations):
            ax = axes2[hi][ii]
            show_surface(all_surfaces[ii][hi], ax=ax, theta=0)
            if hi == 0:
                inc_deg = np.rad2deg(float(inc))
                ax.set_title(f"inc={inc_deg:.1f}°", fontsize=7)
            if ii == 0:
                ax.set_ylabel(harm_labels[hi], fontsize=7)
    fig2.tight_layout()

    plt.show()


def mkcurves(system, t, lmax, y00, ncurves=None, method='pca',
             orbcheck=None, sigorb=None):
    """
    Generates light curves from a star+planet system at times t,
    for positive and negative spherical harmonics with l up to lmax.

    Arguments
    ---------
    system: SurfaceSystem
        A jaxoplanet SurfaceSystem with a star and planet.

    t: 1D array
        Array of times at which to calculate eigencurves.

    lmax: int
        Maximum l to use in spherical harmonic maps.

    y00: 1D array
        Light curve of a normalized, uniform map.

    ncurves: int, optional
        Number of eigencurves to compute. Defaults to all.

    method: str
        PCA method, 'pca' or 'tsvd'.

    Returns
    -------
    eigeny: 2D array
        ncurves x (lmax+1)**2 array of Ylm coefficients for each eigenmap.

    evalues: 1D array
        Eigenvalues from PCA.

    evectors: 2D array
        Eigenvectors from PCA.

    proj: 2D array
        Data projected into the PCA eigenbasis (eigencurves).

    lcs: 2D array
        Raw harmonic light curves before PCA.
    """
    planet_surface = system.body_surfaces[0]
    planet_body = system.bodies[0]
    central = system.central
    central_surface = system.central_surface

    nt = len(t)

    def calcflux(y):
        """Compute planet flux for given full Ylm coefficient array (including Y00)."""
        new_ylm = Ylm.from_dense(y, normalize=False)
        new_planet_surface = Surface(
            y=new_ylm,
            inc=planet_surface.inc,
            period=planet_surface.period,
            radius=planet_surface.radius,
            u=(),
            normalize=False,
            amplitude=1.0,
            phase=planet_surface.phase,
        )
        new_system = SurfaceSystem(central=central, central_surface=central_surface)
        new_system = new_system.add_body(
            period=planet_body.period,
            radius=planet_body.radius,
            mass=planet_body.mass,
            inclination=planet_body.inclination,
            eccentricity=planet_body.eccentricity,
            omega_peri=planet_body.omega_peri,
            time_transit=planet_body.time_transit,
            surface=new_planet_surface,
        )
        flux_result = light_curve(new_system, order=100)(t)
        return flux_result.T[0], flux_result.T[1]

    j_calcflux = jax.jit(calcflux)

    nharm = 2 * ((lmax + 1)**2 - 1)
    lcs = np.zeros((nharm, nt))
    ilc = 0

    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            y = np.zeros((lmax + 1)**2)
            y[0] = 1.0
            y[1 + ilc // 2] = 1.0

            _, pflux = j_calcflux(jnp.array(y))
            lcs[ilc] = np.array(pflux) - y00
            # Negate to get the -1 harmonic without an extra evaluation
            lcs[ilc + 1] = -1.0 * lcs[ilc]
            ilc += 2

    if orbcheck is not None:
        print("Warning: orbcheck not yet implemented for jaxoplanet, skipping...")

    if ncurves is None:
        ncurves = nharm
        if method == 'tsvd':
            ncurves -= 1

    evalues, evectors, proj = pca.pca(lcs, method=method, ncomp=ncurves)
    proj = np.real(proj)

    eigeny = np.zeros((ncurves, (lmax + 1)**2))
    eigeny[:, 0] = 1.0
    for j in range(ncurves):
        yi  = 1
        shi = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                eigeny[j, yi] = evectors.T[j, shi] - evectors.T[j, shi + 1]
                yi  += 1
                shi += 2

    return eigeny, evalues, evectors, proj, lcs


def intensities(fit, data, ln):
    """
    Compute eigenmap intensities at all visible grid cells.

    Arguments
    ---------
    fit: Fit object
    data: Dataset object
    ln: LN object with lmax, ncurves, eigeny attributes.

    Returns
    -------
    intens: 2D array
        (ncurves x nloc) intensities, with uniform-map contribution removed.

    vislat: 1D array
        Visible latitudes in radians.

    vislon: 1D array
        Visible longitudes in radians.
    """
    wherevis = np.where((np.array(fit.lon) + fit.dlon >= data.minvislon) &
                        (np.array(fit.lon) - fit.dlon <= data.maxvislon))

    vislon = jnp.deg2rad(np.array(fit.lon[wherevis].flatten()))
    vislat = jnp.deg2rad(np.array(fit.lat[wherevis].flatten()))
    nloc = len(vislon)

    intens = np.zeros((ln.ncurves, nloc))

    _, ref_planet, _ = initsystem(fit, ln.lmax)
    ref_intensity = np.array(ref_planet.intensity(vislat, vislon))

    def evalintensity(yval):
        new_ylm = Ylm.from_dense(yval, normalize=False)
        new_surface = Surface(
            y=new_ylm,
            inc=ref_planet.inc,
            period=ref_planet.period,
            radius=ref_planet.radius,
            u=(),
            normalize=False,
            amplitude=1.0,
            phase=ref_planet.phase,
        )
        return new_surface.intensity(vislat, vislon)

    evalintensity_jit = jax.jit(evalintensity)

    for k in range(ln.ncurves):
        intens[k] = np.array(evalintensity_jit(jnp.array(ln.eigeny[k]))) - ref_intensity

    return intens, vislat, vislon


def mkmaps(fit, m, ln, params):
    """
    Calculate flux map and brightness temperature map from a single 2D map fit.

    Arguments
    ---------
    fit: Fit object
        Must have fit.lat, fit.lon (in degrees), fit.cfg.

    m: Map object
        Must have m.wlmid, m.filtwl, m.filttrans.

    ln: LN object
        Must have ln.lmax, ln.ncurves, ln.eigeny.

    params: 1D array
        Best-fitting parameters. params[:ncurves] are eigenmap weights,
        params[ncurves] is the uniform component amplitude, and
        params[ncurves+1] is the stellar correction term.

    Returns
    -------
    fmap: 2D array
        Planet flux map, same shape as fit.lat.

    tmap: 2D array
        Brightness temperature map, same shape as fit.lat.
    """
    yval = np.zeros((ln.lmax + 1)**2)
    yval[0] = 1.0
    for j in range(ln.ncurves):
        yval[1:] += params[j] * ln.eigeny[j, 1:]

    _, planet, _ = initsystem(fit, ln.lmax, y=yval)

    fmap = np.array(planet.intensity(np.deg2rad(fit.lat),
                                     np.deg2rad(fit.lon)))

    # Replace default Y00=1 uniform contribution with the fitted amplitude
    fmap += (params[ln.ncurves] - 1.0) / np.pi

    swl   = fit.starwl   if hasattr(fit, 'starwl')   else None
    sspec = fit.starflux if hasattr(fit, 'starflux') else None

    tmap = fmap_to_tmap(fmap, m.wlmid, fit.cfg.planet.r,
                        fit.cfg.star.r, fit.cfg.star.t,
                        params[ln.ncurves + 1],
                        starspec=fit.cfg.star.starspec,
                        fwl=m.filtwl, ftrans=m.filttrans,
                        swl=swl, sspec=sspec)

    return fmap, tmap


def fmap_to_tmap(fmap, meanwl, rp, rs, ts, scorr, starspec='bb',
                 fwl=None, ftrans=None, swl=None, sspec=None,
                 trange=None, fpfs_bb=None):
    '''
    Convert flux map to brightness temperatures.
    See Rauscher et al., 2018, eq. 8

    fmap: 2D array
        Array of star-normalized planet fluxes.

    meanwl: float
        Mean wavelength of planet fluxes, in microns.

    rp: float
        Planet radius. Same units as rs.

    rs: float
        Stellar radius. Same units as rp.

    ts: float
        Stellar temperature in Kelvin.

    scorr: float
        Stellar correction term.

    starspec: str
        'bb', 'bbint', or 'custom'.

    fwl, ftrans: arrays
        Filter wavelengths (microns) and transmission.

    swl, sspec: arrays
        Stellar spectrum wavelengths and spectrum.

    trange, fpfs_bb: arrays
        Pre-computed temperature grid and corresponding filter-integrated
        planet-to-star flux ratios for fast interpolation.
    '''
    meanwl_m = meanwl * 1e-6
    ptemp = (sc.h * sc.c) / (meanwl_m * sc.k)
    sfact = 1 + scorr
    if starspec == 'bb':
        tmap = ptemp / np.log(1 + (rp / rs)**2 *
                              (np.exp(ptemp / ts) - 1) /
                              (np.pi * fmap * sfact))
    elif starspec == 'bbint':
        if fwl is None or ftrans is None:
            print('Must specify filter for integrated blackbody.')
        fwl_m = fwl * 1e-6
        sbb = 2 * sc.h * sc.c**2 / fwl_m**5 / \
            (np.exp(sc.h * sc.c / fwl_m / sc.k / ts) - 1)
        sint = specint(fwl_m, sbb, [fwl_m], [ftrans])
        tmap = ptemp / np.log(1 + (rp / rs)**2 *
                              (2 * sc.h * sc.c**2 / meanwl_m**5) *
                              (1 / np.pi) *
                              (1 / (fmap * sfact)) *
                              (1 / sint))
    elif starspec == 'custom':
        if fwl is None or ftrans is None or sspec is None or swl is None:
            print('Must specify filter and stellar spectrum.')
        if trange is None and fpfs_bb is not None:
            print('Must specify temperatures if supplying fpfs_bb.')
        fwl_m = fwl * 1e-6
        swl_m = swl * 1e-6
        if fpfs_bb is None:
            sspec_int = np.interp(fwl_m, swl_m, sspec)
            trange = np.linspace(50, 5000, 10000)
            bbs = blackbody_wl(trange, fwl_m)
            sspec_fint = np.trapz(ftrans * sspec_int, fwl_m)
            rprs2 = (rp / rs)**2
            fpfs_spec = rprs2 * bbs / sspec_int
            fpfs_bb = np.trapz(fpfs_spec * ftrans * sspec_int,
                               fwl_m, axis=1) / sspec_fint
        interp_fpfs = spi.CubicSpline(fpfs_bb, trange)
        tmap = interp_fpfs(fmap * np.pi)

    return tmap


def specint(wn, spec, filtwn_list, filttrans_list):
    """
    Integrate a spectrum over the given filters.

    Arguments
    ---------
    wn: 1D array
        Wavenumbers (/cm) of the spectrum.

    spec: 1D array
        Spectrum to be integrated.

    filtwn_list: list
        List of arrays of filter wavenumbers, in /cm.

    filttrans_list: list
        List of arrays of filter transmission.

    Returns
    -------
    intspec: 1D array
        The spectrum integrated over each filter.
    """
    if len(filtwn_list) != len(filttrans_list):
        print("ERROR: list sizes do not match.")
        raise Exception

    intspec = np.zeros(len(filtwn_list))

    for i, (filtwn, filttrans) in enumerate(zip(filtwn_list, filttrans_list)):
        idx = np.argsort(filtwn)
        intfunc = spi.interp1d(filtwn[idx], filttrans[idx],
                               bounds_error=False, fill_value=0)
        inttrans = intfunc(wn)
        norminttrans = inttrans / np.trapz(inttrans, wn)
        intspec[i] = np.trapz(spec * norminttrans, wn)

    return intspec


def blackbody_wl(T, wl):
    '''
    Calculates the Planck function for a grid of temperatures and
    wavelengths. Wavelengths must be in m.
    '''
    bb = (2.0 * sc.h * sc.c**2 / (wl[np.newaxis]**5)) \
        * 1 / (np.exp(sc.h * sc.c / wl[np.newaxis] / sc.k / T[:, np.newaxis]) - 1.0)
    return bb


@jit(nopython=True)
def fit_2d(params, ecurves, t, y00, sflux, ncurves, intens, pindex,
           baselines, tlocs, dvecs):
    """
    Basic 2D fitting routine for a single wavelength.

    Arguments
    ---------
    params: 1D float array
        Model parameters: map weights, uniform amplitude, stellar correction,
        normalization factors, detrending coefficients, ramp parameters.

    ecurves: 2D float array
        Eigencurves used as the fitting basis for the planet map.

    t: 1D float array
        All observation times.

    y00: 1D float array
        Uniform-map light curve contribution.

    sflux: 1D float array
        Stellar light curve contribution.

    ncurves: int
        Number of eigencurves.

    intens: 2D float array or None
        Precomputed eigenmap intensities (ncurves x nlocs). When not None,
        used to enforce positive flux at visible locations.

    pindex: 2D bool array
        Boolean index array selecting parameters for each sub-model.

    baselines: tuple of str
        Ramp model names for each visit.

    tlocs: tuple of 1D float arrays
        Local time arrays for each visit.

    dvecs: tuple of 2D float arrays
        Detrending vectors for each visit.
    """
    imodel = 0
    mparams = params[pindex[imodel]]
    imodel += 1

    if intens is not None:
        nloc = intens.shape[1]
        totint = np.zeros(nloc)
        for j in range(nloc):
            totint[j] = np.sum(intens[:, j] * mparams[:ncurves])
            totint[j] += mparams[ncurves] / np.pi
        if np.any(totint <= 0):
            f = np.ones(len(t)) * np.min(totint)
            return f

    f = np.zeros(len(t))

    for i in range(ncurves):
        f += ecurves[i] * mparams[i]

    f += mparams[ncurves] * y00
    f += mparams[ncurves + 1]

    istart = 0
    normparams = params[pindex[imodel]]
    imodel += 1
    for tloc, norm in zip(tlocs, normparams):
        f[istart:istart + len(tloc)] *= norm
        istart += len(tloc)

    f += sflux

    alldvec = np.zeros(len(t))
    istart = 0
    for dvec in literal_unroll(dvecs):
        dmodel = np.ones(dvec.shape[1])
        for j, par in enumerate(params[pindex[imodel]]):
            dmodel += par * dvec[j]
        alldvec[istart:istart + dvec.shape[1]] += dmodel
        istart += dvec.shape[1]
        imodel += 1

    f *= alldvec

    allramp = np.zeros(len(t))
    istart = 0
    for bl, tloc, ipar in zip(baselines, tlocs, pindex[imodel:]):
        rparams = params[ipar]
        if bl == 'none':
            ramp = np.ones(len(tloc))
        elif bl == 'linear':
            ramp = rparams[0] + rparams[1] * tloc
        elif bl == 'quadratic':
            ramp = rparams[0] + rparams[1] * (tloc - rparams[3])**2 + \
                rparams[2] * (tloc - rparams[3])
        elif bl == 'sinusoidal':
            ramp = rparams[0] + rparams[1] * np.sin(
                2 * np.pi * tloc / rparams[2] - rparams[3])
        elif bl == 'exponential':
            ramp = rparams[0] + rparams[1] * np.exp((-rparams[2] * tloc) + rparams[3])
        elif bl == 'linexp':
            ramp = rparams[0] + rparams[1] * tloc + rparams[2] * \
                np.exp((1 / rparams[3]) * -tloc)
        allramp[istart:istart + len(tloc)] += ramp
        istart += len(tloc)

    f *= allramp

    return f


def get_par_2d(fit, d, ln):
    '''
    Returns sensible parameter settings for each 2D model.
    '''
    nmappar = ln.ncurves + 2

    params = np.zeros(nmappar)
    params[ln.ncurves] = 0.001

    pstep = np.ones(nmappar) * 0.01
    pmin  = np.ones(nmappar) * -1.0
    pmax  = np.ones(nmappar) *  1.0

    pstep[ln.ncurves + 1] = 0.0

    pnames   = []
    texnames = []
    for j in range(ln.ncurves):
        pnames.append("C{}".format(j + 1))
        texnames.append("$C_{{{}}}$".format(j + 1))

    pnames.append("C0")
    texnames.append("$C_0$")
    pnames.append("scorr")
    texnames.append("$s_{corr}$")

    nnormpar = len(d.visits)
    params   = np.concatenate((params,   np.repeat(1.0, nnormpar)))
    pmin     = np.concatenate((pmin,     np.repeat(0.8, nnormpar)))
    pmax     = np.concatenate((pmax,     np.repeat(1.2, nnormpar)))
    pnames   = np.concatenate((pnames,   ['N{}'.format(i) for i in range(1, nnormpar + 1)]))
    texnames = np.concatenate((texnames, ['$N_{}$'.format(i) for i in range(1, nnormpar + 1)]))
    for v in d.visits:
        pstep = np.concatenate((pstep, (0.01,) if v.renormalize else (0.0,)))

    ndvecpar = []
    for v in d.visits:
        if v.detrend:
            npar = v.dvec.shape[0]
            params   = np.concatenate((params,   np.repeat(0.0, npar)))
            pstep    = np.concatenate((pstep,    np.repeat(0.1, npar)))
            pmin     = np.concatenate((pmin,     np.repeat(-np.inf, npar)))
            pmax     = np.concatenate((pmax,     np.repeat( np.inf, npar)))
            pnames   = np.concatenate((pnames,   ['d{}'.format(i) for i in range(1, npar + 1)]))
            texnames = np.concatenate((texnames, ['$d_{}$'.format(i) for i in range(1, npar + 1)]))
        else:
            npar = 0
        ndvecpar.append(npar)

    nramppar = []
    for v in d.visits:
        if v.baseline == 'none':
            npar = 0
        elif v.baseline == 'linear':
            params   = np.concatenate((params,   (1.0, 0.0)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001)))
            pmin     = np.concatenate((pmin,     (0.8, -np.inf)))
            pmax     = np.concatenate((pmax,     (1.2,  np.inf)))
            pnames   = np.concatenate((pnames,   ('b', 'm')))
            texnames = np.concatenate((texnames, ('$b$', '$m$')))
            npar = 2
        elif v.baseline == 'quadratic':
            params   = np.concatenate((params,   (1.0, 0.0,  0.0,  0.0)))
            pstep    = np.concatenate((pstep,    (0.01, 0.01, 0.01, 0.0)))
            pmin     = np.concatenate((pmin,     (0.8, -1.0, -1.0, -np.inf)))
            pmax     = np.concatenate((pmax,     (1.2,  1.0,  1.0,  np.inf)))
            pnames   = np.concatenate((pnames,   ('r0', 'r1', 'r2', 't0')))
            texnames = np.concatenate((texnames, ('r_0', '$r_1$', '$r_2$', '$t_0$')))
            npar = 3
        elif v.baseline == 'sinusoidal':
            params   = np.concatenate((params,   (1.0, -3.6e-5, 0.0885, 2.507)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001, 0.001, 0.1)))
            pmin     = np.concatenate((pmin,     (0.8, -1.0, 0.05, -np.pi)))
            pmax     = np.concatenate((pmax,     (1.2,  1.0, 0.15,  np.pi)))
            pnames   = np.concatenate((pnames,   ('b', 'Amp.', 'Period', 'Phase')))
            texnames = np.concatenate((texnames, ('$b$', 'Amp.', 'Period', 'Phase')))
            npar = 4
        elif v.baseline == 'exponential':
            params   = np.concatenate((params,   (1.0, 0.00001, 0.00001, 0.00001)))
            pstep    = np.concatenate((pstep,    (0.01, 0.01, 0.01, 0.01)))
            pmin     = np.concatenate((pmin,     (0.8, -5, -5, -5)))
            pmax     = np.concatenate((pmax,     (1.2, 30, 30, 30)))
            pnames   = np.concatenate((pnames,   ('r0', 'r1', 'r2', 'r3')))
            texnames = np.concatenate((texnames, ('$r_0$', '$r_1$', '$r_2$', '$r_3$')))
            npar = 4
        elif v.baseline == 'linexp':
            params   = np.concatenate((params,   (1.0, -0.00219881, 0.00010304, 0.01629347)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001, 0.001, 0.001)))
            pmin     = np.concatenate((pmin,     (0.8, -1, -0.01, 0.0)))
            pmax     = np.concatenate((pmax,     (1.2,  1,  0.01, 0.2)))
            pnames   = np.concatenate((pnames,   ('b', 'm', 'A', 'tau')))
            texnames = np.concatenate((texnames, ('$b$', '$m$', '$A$', '$\\tau$')))
            npar = 4
        else:
            print("Unrecognized baseline model.")
            sys.exit()
        nramppar.append(npar)

    npars   = [nmappar, nnormpar] + ndvecpar + nramppar
    nparams = len(params)
    pindex  = np.zeros((len(npars), nparams))
    for i, npar in enumerate(npars):
        start = int(np.sum(npars[:i]))
        pindex[i, start:start + npar] = True
    pindex = pindex.astype(bool)

    return params, pstep, pmin, pmax, pnames, texnames, pindex


def mkcurves_vary(system, t, lmax, y00, ncurves=None, method='pca', obl=0, p_inc=0,
                  orbcheck=None, sigorb=None):
    """
    Generates light curves from a star+planet system at times t,
    for positive and negative spherical harmonics with l up to lmax,
    with a specified obliquity and planet inclination.

    Arguments
    ---------
    system: SurfaceSystem
        A jaxoplanet SurfaceSystem with a star and planet.

    t: 1D array
        Array of times at which to calculate eigencurves.

    lmax: int
        Maximum l to use in spherical harmonic maps.

    y00: 1D array
        Light curve of a normalized, uniform map.

    obl: float
        Obliquity of the planet in radians.

    p_inc: float
        Planet surface inclination in radians.

    Returns
    -------
    eigeny, evalues, evectors, proj, lcs
        Same format as mkcurves.
    """
    planet_surface = system.body_surfaces[0]
    planet_body = system.bodies[0]
    central = system.central
    central_surface = system.central_surface

    nt = len(t)

    def evalflux(yval):
        n_coeffs = (lmax + 1)**2
        ylm_coeffs = jnp.zeros(n_coeffs)
        ylm_coeffs = ylm_coeffs.at[0].set(1.0)
        ylm_coeffs = ylm_coeffs.at[1:].set(yval)
        new_ylm = Ylm.from_dense(ylm_coeffs, normalize=True)
        new_planet_surface = Surface(
            y=new_ylm,
            inc=p_inc,
            period=planet_surface.period,
            radius=planet_surface.radius,
            u=(),
            normalize=False,
            amplitude=1.0,
            phase=planet_surface.phase,
            obl=obl,
        )
        new_system = SurfaceSystem(central=central, central_surface=central_surface)
        new_system = new_system.add_body(
            period=planet_body.period,
            radius=planet_body.radius,
            mass=planet_body.mass,
            inclination=planet_body.inclination,
            eccentricity=planet_body.eccentricity,
            omega_peri=planet_body.omega_peri,
            time_transit=planet_body.time_transit,
            surface=new_planet_surface,
        )
        flux_result = light_curve(new_system, order=100)(t)
        return np.array(flux_result.T[0]), np.array(flux_result.T[1])

    nharm = 2 * ((lmax + 1)**2 - 1)
    lcs = np.zeros((nharm, nt))
    ilc = 0

    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            yval = np.zeros(nharm // 2)
            yval[ilc // 2] = 1.0
            _, lcs[ilc] = evalflux(yval)
            lcs[ilc + 1] = lcs[ilc].copy()
            lcs[ilc + 1] -= y00
            ilc += 2

    if orbcheck is not None:
        print("Warning: orbcheck not yet implemented for jaxoplanet, skipping...")

    lcs -= y00

    if ncurves is None:
        ncurves = nharm
        if method == 'tsvd':
            ncurves -= 1

    evalues, evectors, proj = pca.pca(lcs, method=method, ncomp=ncurves)
    proj = np.real(proj)

    eigeny = np.zeros((ncurves, (lmax + 1)**2))
    eigeny[:, 0] = 1.0
    for j in range(ncurves):
        yi  = 1
        shi = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                eigeny[j, yi] = evectors.T[j, shi] - evectors.T[j, shi + 1]
                yi  += 1
                shi += 2

    return eigeny, evalues, evectors, proj, lcs


def main():
    fit = dummyfit.create_dummy_fit()
    star_surface, planet_surface, system = initsystem(fit, 3)


main()
