#!/usr/bin/env python3
"""
Plots eigencurves and surface maps across different obliquity and inclination values.
"""


# Try changing the inclination and keep obliquity constant () 
# make a gif with planet revoling around star 

import sys
import os

import matplotlib


import jax.numpy as jnp
from jaxoplanet.starry.light_curves import light_curve
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.starry.visualization import show_surface
import numpy as np
import time
import matplotlib.pyplot as plt

# Add lib directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitclass as fc
import helper as utils_jax  # This has our jaxoplanet initsystem
#from lib import model_2d as model  # Use model_2d to avoid theano dependencies

def plot_obliquity_inclination(cfile):
    """
    Simplified version of map2d that:
    1. Reads the configuration file
    2. Reads the data
    3. Initializes the jaxoplanet system
    4. Computes planet and star positions at observation times

    Arguments
    ---------
    cfile : str
        Path to configuration file

    Returns
    -------
    fit : Fit object
        Fit object with loaded data and computed positions
    system : SurfaceSystem
        Jaxoplanet system object
    """

    # Create the master fit object
    fit = fc.Fit()

    # Set up logging for jaxoplanet comparisons
    log_file = os.path.join(os.path.dirname(cfile), "jaxoplanet_outputs.log")
    print(f"Logging jaxoplanet outputs to: {log_file}")

    print("="*60)
    print("SIMPLIFIED MAP2D WITH JAXOPLANET")
    print("="*60)

    print("\n[1/5] Reading the configuration file...")
    t_start = time.time()
    fit.read_config(cfile)
    cfg = fit.cfg
    t_elapsed = time.time() - t_start
    print(f"  ✓ Config loaded from: {cfile} (took {t_elapsed:.3f}s)")
    print(f"  ✓ Star: M={cfg.star.m} Msun, R={cfg.star.r} Rsun")
    print(f"  ✓ Planet: M={cfg.planet.m} Msun, R={cfg.planet.r} Rsun")
    print(f"  ✓ Planet orbit: P={cfg.planet.porb} days, inc={cfg.planet.inc} deg")

    print("\n[2/5] Reading the data...")
    t_start = time.time()
    fit.read_data()
    t_elapsed = time.time() - t_start
    print(f"  ✓ Number of datasets: {len(fit.datasets)} (took {t_elapsed:.3f}s)")
    for i, d in enumerate(fit.datasets):
        print(f"  ✓ Dataset {i+1}: {d.name}")
        print(f"    - Number of visits: {len(d.visits)}")
        total_times = sum(len(v.t) for v in d.visits)
        print(f"    - Total observation times: {total_times}")

    print("\n[3/5] Reading filters...")
    t_start = time.time()
    fit.read_filters()
    t_elapsed = time.time() - t_start
    print(f"  ✓ Filter mean wavelengths (μm): (took {t_elapsed:.3f}s)")
    for d in fit.datasets:
        for i, wl in enumerate(d.wlmid):
            print(f"    - Filter {i+1}: {wl:.3f} μm")

    print("\n[4/5] Initializing jaxoplanet system...")
    t_start = time.time()
    # Use lmax=1 for simplicity (can change later)
    star_surface, planet_surface, system = utils_jax.initsystem(fit, ydeg=1)
    t_elapsed = time.time() - t_start
    print(f"  ✓ Star surface created: (took {t_elapsed:.3f}s)")
    print(f"    - Radius: {star_surface.radius} Rsun")
    print(f"    - Period: {star_surface.period} days")
    print(f"  ✓ Planet surface created:")
    print(f"    - Radius: {planet_surface.radius} Rsun")
    print(f"    - Period: {planet_surface.period} days")
    print(f"    - Ylm degree: {planet_surface.y.deg}")
    print(f"  ✓ System created with {len(system.bodies)} planet(s)")



    print("\n" + "="*60)
    print("Calculating uniform-map planet and star fluxes...")
    t_start = time.time()
    for d in fit.datasets:
        # Compute light curve with uniform map (Y00 only)
        # The system is already initialized with Y00=1.0 uniform map
        flux_result = light_curve(system)(d.t)
        d.sflux = np.array(flux_result.T[0])      # Star flux (convert to numpy for numba)
        d.pflux_y00 = np.array(flux_result.T[1])  # Planet flux (convert to numpy for numba)
        print(f"  ✓ Dataset {d.name}: computed {len(d.t)} flux points")
    t_elapsed = time.time() - t_start
    print(f"  Total time for uniform flux calculation: {t_elapsed:.3f}s")


    print("\n" + "="*60)
    print("Optimizing 2D maps.")
    for d in fit.datasets:
        d.maps = []
        for i in range(len(d.wlmid)):
            print("{:.2f} um".format(d.wlmid[i]))
            m = fc.Map()

            d.maps.append(m)

            m.wlmid     = d.wlmid[i]
            m.filtwl    = d.filtwl[i]
            m.filtwn    = d.filtwn[i]
            m.filttrans = d.filttrans[i]
            m.flux      = d.flux[i]
            m.ferr      = d.ferr[i]

            m.subdir = '{}-filt{}'.format(d.name, i + 1)
            os.makedirs(os.path.join(cfg.twod.outdir, m.subdir), exist_ok=True)


            for l in range(1, cfg.twod.lmax+1):
                for n in range(0, cfg.twod.ncurves+1):
                    # Skip cases where n is higher than the number of
                    # available eigencurves, which is (l+1)**2, minus
                    # the uniform (l=0) case, since that's included by
                    # default
                    if n > (l+1)**2 - 1:
                        continue

                    # Also let's only do the n=0 case once, since
                    # it's exactly the same fit for every lmax.
                    # Link the LN objects for looping simplicity later
                    if l > 1 and n==0:
                        setattr(m, 'l{}n{}'.format(l, n), m.l1n0)
                        continue

                    print("Fitting lmax={}, n={}".format(l,n))
                    setattr(m, 'l{}n{}'.format(l, n), fc.LN())
                    ln = getattr(m, 'l{}n{}'.format(l, n))

                    ln.subdir = 'l{}n{}'.format(l,n)

                    ln.wlmid = d.wlmid[i]

                    ln.ncurves = n
                    ln.lmax    = l

                    # New planet object with updated lmax
                    print("    Initializing system with lmax={}...".format(l))
                    t_init_start = time.time()
                    star_surface, planet_surface, system_ln = utils_jax.initsystem(fit, ln.lmax)
                    t_init_elapsed = time.time() - t_init_start
                    print("      System initialization took {:.3f}s".format(t_init_elapsed))

                    print("    Running PCA to determine eigencurves...")
                    ncomp = ln.ncurves
                    if ln.ncurves == 0:
                        ncomp = None
                    if(n < 2 or l < 2 ):
                        continue 
                    p_inc = 0
                    planet_surface_ln = system_ln.body_surfaces[0]
                    n_coeffs = (ln.lmax + 1) ** 2

                    results = []
                    for oi in range(10):
                        obl = (oi / 9) * 3.14159
                        obl_deg = obl * (180 / 3.14159)
                        print("lmax=" + str(ln.lmax) + f"  obl={obl_deg:.1f}°")

                        ln.eigeny, ln.evalues, ln.evectors, ln.ecurves, ln.lcs = \
                        utils_jax.mkcurves_vary(system_ln, d.t, ln.lmax,
                                          d.pflux_y00, ncurves=ncomp,
                                          method=cfg.twod.pca,
                                          orbcheck=cfg.twod.orbcheck,
                                          sigorb=cfg.twod.sigorb, p_inc=p_inc, obl=float(obl))

                        ylm_coeffs = jnp.array(ln.eigeny[1]) if len(ln.eigeny) > 0 else jnp.zeros(n_coeffs)
                        new_ylm = Ylm.from_dense(ylm_coeffs, normalize=False)
                        surf = Surface(
                            y=new_ylm,
                            inc=planet_surface_ln.inc,
                            period=planet_surface_ln.period,
                            radius=planet_surface_ln.radius,
                            u=(),
                            normalize=False,
                            amplitude=1.0,
                            phase=planet_surface_ln.phase,
                            obl=float(obl),
                        )
                        results.append((obl_deg, ln.ecurves.copy(), surf))

                    fig1, axes1 = plt.subplots(2, 5,
                               figsize=(4 * 5, 3 * 2), squeeze=False)
                    fig1.suptitle(f" lmax = {l} | n curves = {n} | Light Curves vs Obliquity  |  p_inc={p_inc * (180/3.14159):.1f}°")
                    fig2, axes2 = plt.subplots(2, 5,
                               figsize=(3 * 5, 3 * 2), squeeze=False)
                    fig2.suptitle(f" lmax = {l} | n curves = {n} | Obliquities vs Surface  |  p_inc={p_inc * (180/3.14159):.1f}°")

                    for oi, (obl_deg, ecurves, surf) in enumerate(results):
                        ax = axes1[oi // 5][oi % 5]
                        for ci in range(len(ecurves)):
                            ax.plot(d.t, ecurves[ci])
                        ax.set_title(f"obl={obl_deg:.1f}°")
                        ax.set_xlabel("Time (days)")
                        ax.set_ylabel("ΔFlux")

                        ax2 = axes2[oi // 5][oi % 5]
                        show_surface(surf, ax=ax2, theta=0)
                        ax2.set_title(f"obl={obl_deg:.1f}°")

                    fig1.tight_layout()
                    fig2.tight_layout()
                    plt.show()

                    obl = 0
                    inc_results = []
                    for ii in range(10):
                        p_inc = (ii / 9) * 3.14159
                        inc_deg = p_inc * (180 / 3.14159)
                        print("lmax=" + str(ln.lmax) + f"  inc={inc_deg:.1f}°")

                        ln.eigeny, ln.evalues, ln.evectors, ln.ecurves, ln.lcs = \
                        utils_jax.mkcurves_vary(system_ln, d.t, ln.lmax,
                                          d.pflux_y00, ncurves=ncomp,
                                          method=cfg.twod.pca,
                                          orbcheck=cfg.twod.orbcheck,
                                          sigorb=cfg.twod.sigorb, p_inc=float(p_inc), obl=obl)

                        ylm_coeffs = jnp.array(ln.eigeny[1]) if len(ln.eigeny) > 0 else jnp.zeros(n_coeffs)
                        new_ylm = Ylm.from_dense(ylm_coeffs, normalize=False)
                        surf = Surface(
                            y=new_ylm,
                            inc=float(p_inc),
                            period=planet_surface_ln.period,
                            radius=planet_surface_ln.radius,
                            u=(),
                            normalize=False,
                            amplitude=1.0,
                            phase=planet_surface_ln.phase,
                            obl=obl,
                        )
                        inc_results.append((inc_deg, ln.ecurves.copy(), surf))

                    fig3, axes3 = plt.subplots(2, 5,
                               figsize=(4 * 5, 3 * 2), squeeze=False)
                    fig3.suptitle(f" lmax = {l} | n curves = {n} | Light Curves vs Inclination  |  obl={obl * (180/3.14159):.1f}°")
                    fig4, axes4 = plt.subplots(2, 5,
                               figsize=(3 * 5, 3 * 2), squeeze=False)
                    fig4.suptitle(f" lmax = {l} | n curves = {n} | Inclination vs Surface  |  obl={obl * (180/3.14159):.1f}°")

                    for ii, (inc_deg, ecurves, surf) in enumerate(inc_results):
                        ax = axes3[ii // 5][ii % 5]
                        for ci in range(len(ecurves)):
                            ax.plot(d.t, ecurves[ci])
                        ax.set_title(f"inc={inc_deg:.1f}°")
                        ax.set_xlabel("Time (days)")
                        ax.set_ylabel("ΔFlux")

                        ax4 = axes4[ii // 5][ii % 5]
                        show_surface(surf, ax=ax4, theta=0)
                        ax4.set_title(f"inc={inc_deg:.1f}°")

                    fig3.tight_layout()
                    fig4.tight_layout()
                    plt.show()

                    continue


if __name__ == "__main__":
    # Check if config file provided
    if len(sys.argv) < 2:
        print("Usage: python map2d_jax_simple.py <config_file>")
        print("\nExample:")
        print("  python map2d_jax_simple.py ../wasp76-example.cfg")
        sys.exit(1)

    cfile = sys.argv[1]

    # Check if file exists
    if not os.path.exists(cfile):
        print(f"Error: Config file not found: {cfile}")
        sys.exit(1)

    # Run simplified map2d
    plot_obliquity_inclination(cfile)

    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nVariables available:")
    print("  fit    - Fit object with loaded data")
    print("  system - Jaxoplanet SurfaceSystem object")
    print("\nAccess data:")
    print("  fit.datasets[0].x  - X positions")
    print("  fit.datasets[0].y  - Y positions")
    print("  fit.datasets[0].z  - Z positions")
    print("  fit.datasets[0].t  - Observation times")
    print("\nAccess system:")
    print("  system.bodies[0]         - Planet body")
    print("  system.body_surfaces[0]  - Planet surface")
    print("  system.central           - Star")
    print("  system.central_surface   - Star surface")


# TODO: 
# - Fix up plots so that it generates ... 
# - Give it fake data to see the effects of inclination after running Theresa
#       - Get light curves from jaxoplanet
# - Add white noise and inject fake data and then run into theresa 
# - Look at with inclincation vs. without inclination
#    - Maybe look at real data and looking at existing publications


# - Synthetic data injection 
# Create python script for the above ^^
# Play around with hotspot (making hotspot) (different harmonic weights)
    # play around do it based on intution 
    