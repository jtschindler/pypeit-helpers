#!/usr/bin/env python3

import os
import argparse
import sort_gnirs

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
            Automatic preparation of GNIRS data for PypeIT. This 
            routine works on the 'raw' downloaded GNRIS files. It extracts
            them, deletes unncessary files, copies them,
            and runs pypeit_setup. 
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-o', '--object_name', required=True, type=str,
                        help='')

    parser.add_argument('-d', '--datadir', required=False, type=str,
                        help='Path to the directory with the downloaded '
                             'GNIRS data.')

    parser.add_argument('-v', '--verbosity', required=False, type=int,
                        help='Parameter to set verbosity. It is advised to '
                             'set verbosity = 1.')

    parser.add_argument('-r', '--remove', required=False, type=bool,
                        help='Boolean to indicate whether the original files '
                             'should be removed.')

    parser.add_argument('-m', '--mode', required=False, type=str,
                        help='Set mode to "data"= data preparation only, '
                             '"setup" = run setup files only, or "clean" = '
                             'provide clean setup tables only')

    parser.add_argument('--deltamjd', required=False, type=str,
                        help='Set the value for the maximum difference in '
                             'mjds, '
                             'which sets the association of bias, arc and '
                             'pixelflats to the science "calib" file. The '
                             'default value is 0.65')

    parser.add_argument('--sortby', required=False, type=str,
                        help='String to reference either the "mjd" or the '
                             '"filename" by which observation files are '
                             'sorted. If the mjd header information is '
                             'incorrect the filename might be better to use.')

    # parser.add_argument('-std', '--standard', required=False, type=str,
    #                     default=False, help='Boolean to consider to use or '
    #                                         'delete the standard files.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    if args.object_name is not None:
        obj_name = args.object_name
    else:
        obj_name = 'UNNAMED_OBJECT'
    if args.datadir is not None:
        path = args.datadir
    else:
        path = os.getcwd()+'/'
    if args.remove is not None:
        remove = args.remove
    else:
        remove = False
    if args.verbosity is not None:
        verbosity = args.verbosity
    else:
        verbosity = 1
    if args.mode is not None:
        mode = args.mode
    else:
        mode = None
    if args.deltamjd is not None:
        delta_mjd = args.deltamjd
    else:
        delta_mjd = 0.65

    if args.sortby is not None:
        sortby = args.sortby
    else:
        sortby = 'mjd'

    sort_gnirs.prepare_gnirs_data(path, obj_name,
                                        remove_originals=remove,
                                        verbosity=verbosity,
                                        mode=mode,  delta_mjd=delta_mjd,
                                  sortby=sortby)
