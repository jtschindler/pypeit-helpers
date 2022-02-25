
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from pypeit import specobj
from pypeit import specobjs

import numpy as np

# Define colors
black = (0, 0, 0)
orange = (230/255., 159/255., 0)
blue = (86/255., 180/255., 233/255.)
green = (0, 158/255., 115/255.)
yellow = (240/255., 228/255., 66/255.)
dblue = (0, 114/255., 178/255.)
vermillion = (213/255., 94/255., 0)
purple = (204/255., 121/255., 167/255.)

def get_plot_ylim_specobj(sobj):

    if sobj['OPT_FLAM'] is not None:
        flux = sobj['OPT_FLAM'][sobj['OPT_MASK']]
    elif sobj['OPT_COUNTS'] is not None:
        flux = sobj['OPT_COUNTS'][sobj['OPT_MASK']]
    else:
        raise ValueError('[ERROR] Could not find COUNTS or FLAM in SpecObj')

    return get_plot_ylim(flux)


def get_plot_ylim(flux):

    percentiles = np.percentile(flux, [16, 84])
    median = np.median(flux)

    ylim_min = -0.5 * median
    ylim_max = 4 * percentiles[1]

    return [ylim_min, ylim_max]


def smooth_flux(flux, width=10, kernel="boxcar"):

    if kernel == "boxcar" or kernel == "Boxcar":
        kernel = Box1DKernel(width)
    elif kernel == "gaussian" or kernel == "Gaussian":
        kernel = Gaussian1DKernel(width)

    return convolve(flux, kernel)


def read_pypeit_specobjs(filename):

    specobjects = specobjs.SpecObjs
    return specobjects.from_fitsfile(filename)


def plot_pypeit_specobjs(specobjects, smooth=None, ymin=None, ymax=None):

    # Set up plot
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=(15, 7),
                           dpi=140)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

    # Plot the PypeIt Specobjs files
    ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle=':',
               label='Line of 0 flux density')

    for idx, sobj in enumerate(specobjects):
        if idx % 2 == 0:
            color = vermillion
        else:
            color = dblue

        mask = sobj['OPT_MASK']
        if 'OPT_FLAM' in sobj.__dict__.keys() and sobj['OPT_FLAM'] is not None:
            flux = sobj['OPT_FLAM']
            ax.set_ylabel(r'Flux density ($10^{-17}\rm{erg}/\rm{s}\rm{cm}^2/\rm{\AA}$)')
        elif 'OPT_COUNTS' in sobj.__dict__.keys() and sobj['OPT_COUNTS'] is \
                not None:
            flux = sobj['OPT_COUNTS']
            ax.set_ylabel(r'Counts')
        else:
            raise ValueError('[ERROR] Could not find COUNTS or FLAM in SpecObj')

        if isinstance(smooth, int):
            flux = smooth_flux(flux, smooth)

        ax.plot(sobj['OPT_WAVE'][mask], flux[mask], color=color, lw=1.5, alpha=0.5)

    # Set plot limits
    ylims = get_plot_ylim_specobj(sobj)
    if ymin is None and ymax is None:
        ax.set_ylim(ylims)
    elif ymin is None and ymax is not None:
        ax.set_ylim(ylims[0], ymax)
    elif ymax is None and ymin is not None:
        ax.set_ylim(ymin, ylims[1])

    ax.set_xlabel(r'Wavelength ($\rm{\AA}$)')



    plt.legend()
    plt.show()


def plot_pypeit_onespec(hdu, smooth=None, ymin=None, ymax=None,
                        comparison=None):
    # Set up plot
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=(15, 7),
                           dpi=140)
    fig.subplots_adjust(left=0.09, right=0.92, top=0.89, bottom=0.16)

    # Plot the PypeIt Specobjs files
    ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle=':',
               label='Line of 0 flux density')

    spec = hdu[1].data


    mask = np.array(spec['mask'], dtype=bool)
    flux = spec['flux']
    ivar = spec['ivar']

    if 'telluric' in hdu[1].data.columns.names:
        ax2 = ax.twinx()

        ax2.plot(spec['wave'][mask], spec['telluric'][mask], color=dblue,
                 label='telluric model', zorder=0)
        ax2.axhline(y=00, lw=1.5, color=dblue, ls=':', zorder=0)
        ax2.set_ylim(-1, 1)

        ax2.set_ylabel('Telluric model')
        ax2.legend()
        ax.set_zorder(ax2.get_zorder() + 1)

    if isinstance(smooth, int):
        flux = smooth_flux(flux, smooth)

    ax.plot(spec['wave'][mask], flux[mask], color='k', lw=1.5, label='flux')

    ax.plot(spec['wave'][mask], 1/np.sqrt(ivar[mask]), color='0.5', lw=3,
            label='flux error')

    if 'obj_model' in hdu[1].data.columns.names:
        ax.plot(spec['wave'][mask], spec['obj_model'][mask], color=vermillion,
                lw=2, label='object model')

    if comparison is not None:
        hdu2 = fits.open(comparison)
        if hdu2[1].header['DMODCLS'] == 'OneSpec':

            spec2 = hdu2[1].data
            mask2 = np.array(spec2['mask'], dtype=bool)
            ax.plot(spec2['wave'][mask2], spec2['flux'][mask2], color=green,
                    lw=1.5, label='comparison spectrum')

    # Set plot limits
    ylims = get_plot_ylim(spec['flux'][mask])
    if ymin is None and ymax is None:
        ax.set_ylim(ylims)
    elif ymin is None and ymax is not None:
        ax.set_ylim(ylims[0], ymax)
    elif ymax is None and ymin is not None:
        ax.set_ylim(ymin, ylims[1])

    ax.set_xlabel(r'Wavelength ($\rm{\AA}$)')
    ax.set_ylabel(r'Flux density ($10^{-17}\rm{erg}/\rm{s}\rm{cm}^2/\rm{\AA}$)')


    ax.patch.set_visible(False)  # hide the 'canvas'

    ax.legend()

    plt.show()


def plot_pypeit_spectrum(filename, smooth=None, ymin=None, ymax=None,
                         comparison=None):

    hdu = fits.open(filename)

    if hdu[1].header['DMODCLS'] == 'SpecObj':
        sobjs = read_pypeit_specobjs(filename)
        plot_pypeit_specobjs(sobjs, smooth=smooth, ymin=ymin, ymax=ymax)
    elif hdu[1].header['DMODCLS'] == 'OneSpec':
        plot_pypeit_onespec(hdu, smooth=smooth, ymin=ymin, ymax=ymax,
                            comparison=comparison)
    else:
        raise ValueError('[ERROR] PypeIt spectrum datatype not understood.')

