#!/usr/bin/env python3


import os
import argparse


from configobj import ConfigObj

from pypeit.par.util import parse_pypeit_file
from pypeit import msgs

def inspect_reduction_files(pypeit_file):

    print(len(parse_pypeit_file(pypeit_file, runtime=True)))

    print(parse_pypeit_file(pypeit_file, runtime=True))

    cfg_lines, data_files, frametype, usrdata, setups, setup_dict \
        = parse_pypeit_file(pypeit_file, runtime=True)

    cfg = ConfigObj(cfg_lines)
    spectrograph_name = cfg['rdx']['spectrograph']

    # Sort data files, so that uncleaned and clean files will be opened after
    # another.
    data_files.sort(key=lambda x: x.split('/')[-1][-9:-5])

    for idx, file in enumerate(data_files):

        os.system('pypeit_view_fits {} {} --proc'.format(spectrograph_name,
                                                         file))


        filetype = usrdata[idx]['frametype']
        target = usrdata[idx]['target']
        msgs.info("Filetype {}; Target {}; File {}".format(filetype, target,
                                                         file))

        if file.split('/')[-1][0] != 'c':

            clean_file = '/' + os.path.join(*file.split('/')[:-1]) +\
                         '/c' + file.split('/')[-1]

            question = 'Do you want to display the cleaned datafile?'

            reply = str(input(question + ' (y/n): ')).lower().strip()
            if reply[:1] == 'y':
                if os.path.isfile(clean_file):
                    os.system(
                        'pypeit_view_fits {} {} --proc'.format(spectrograph_name,
                                                                clean_file))
                    msgs.info(
                        "Opened cleaned file {}".format(clean_file))
                else:
                    msgs.info('Cleaned file {} does not exist'.format(clean_file))
                    print(os.path.isfile(clean_file))

                input("Press any key to continue")





def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
            Inspect the fits images for the reduction with ginga using 
            pypeit_view_fits iteratively. 
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('pypeit_file', type=str,
                        help='Pypeit file')


    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    inspect_reduction_files(args.pypeit_file)