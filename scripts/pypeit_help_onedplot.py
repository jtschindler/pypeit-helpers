#!/usr/bin/env python3


import argparse
import plot

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
            Quick plotting routine for 1D spectra using pypeit-helpers. 
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('filename', type=str,
                        help='Filename of spectrum to plot')

    parser.add_argument('-s', '--smooth', required=False, type=int,
                        help='Number of pixels for a simple boxcar smoothing '
                             'of the spectrum.',
                        default=None)

    parser.add_argument('-ymax', '--ymax', required=False, type=float,
                        help='Setting the maximum value for the y-axis.',
                        default=None)

    parser.add_argument('-ymin', '--ymin', required=False, type=float,
                        help='Setting the maximum value for the y-axis.',
                        default=None)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    print(args.smooth)

    plot.plot_pypeit_spectrum(args.filename,
                              smooth=args.smooth,
                              ymin=args.ymin,
                              ymax=args.ymax)

