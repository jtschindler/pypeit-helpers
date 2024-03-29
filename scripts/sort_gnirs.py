#!/usr/bin/env python3

import os
import glob
import numpy as np

import shutil
import bz2
import tarfile

import datetime

import pandas as pd
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table

from distutils.dir_util import copy_tree

template_signature_to_delete = ['LgPinholes_G5530']

obj_name_to_delete = ['LAMP,FMTCHK', 'LAMP,ORDERDEF', 'LAMP,AFC']

decker_to_delete = ['Acq_G5530']


def read_sorted_file(filename):
    with open(filename) as f:
        content = f.readlines()

    output = open('temp.txt', 'w+')
    for line in content:
        if '|' in line:
            output.write(line)
    output.close()

    df = read_sorted_table('temp.txt')

    os.remove('temp.txt')
    return df


def read_sorted_table(filename):
    table = ascii.read(filename, delimiter='|', format='fixed_width')
    df = table.to_pandas()

    return df


def write_sorted_table(df, filename, sortby='mjd'):
    df.sort_values(by=[sortby], inplace=True)

    table = Table.from_pandas(df)
    ascii.write(table, filename, delimiter='|', format='fixed_width',
                overwrite=True)



def untar_files(path, verbosity, remove_originals):

    files_to_untar = glob.glob(path + '*.tar')
    if verbosity > 0:
        print('[INFO] Untaring all *.tar files in path')
    for file in files_to_untar:
        if verbosity > 2:
            print('[INFO] Untaring {}'.format(file))
        tf = tarfile.open(file)
        tf.extractall()

        if remove_originals:
            os.remove(file)


def unzip_files_bz2(path, verbosity, remove_originals):

    files_to_unzip = glob.glob(path + '*.fits.bz2')
    if verbosity > 0:
        print('[INFO] Unzipping all *.fits.zip files in path')
    for filename in files_to_unzip:
        if verbosity > 2:
            print('[INFO] Unzipping {}'.format(filename))

        newfilename = filename[:-4]
        with open(newfilename, 'wb') as new_file, open(filename,
                                                       'rb') as file:
            decompressor = bz2.BZ2Decompressor()
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(decompressor.decompress(data))

        if remove_originals:
            os.remove(filename)


def make_image_df_gnirs(datapath, save=False, save_name=None, verbosity=0,
                        sortby='mjd'):
    """

    :param datapath:
    :param save:
    :param save_name:
    :return:
    """
    filename_list = []
    obsid_list = []
    objname_list = []
    programid_list = []
    instrument_list = []
    date_list = []
    mjd_list = []
    exptime_list = []

    obs_class_list = []
    obs_type_list = []
    grating_list = []
    filter1_list = []
    filter2_list = []

    fits_list = glob.glob('{}/*.fits'.format(datapath))

    for file in fits_list:

        if verbosity > 2:
            print('[INFO] Reading fits header of {}'.format(file))

        hdu = fits.open(file)
        hdr = hdu[0].header

        filename_list.append(file)
        obsid_list.append(hdr['OBSID'])  # Observation ID
        objname_list.append(hdr['OBJECT'])  # Original target
        programid_list.append(hdr['GEMPRGID'])  # Gemini program ID
        instrument_list.append(hdr['INSTRUME'])
        date_list.append(hdr['DATE-OBS'])
        mjd_list.append(hdr['MJD_OBS'])
        exptime_list.append(hdr['EXPTIME'])

        obs_class_list.append(hdr['OBSCLASS'])  # Observation class
        obs_type_list.append(hdr['OBSTYPE'])  # Observation type
        grating_list.append(hdr['GRATING'])  # Grating
        filter1_list.append(hdr['FILTER1'])  # Filter 1
        filter2_list.append(hdr['FILTER2'])  # Filter 2

    columns = ['filename', 'obs_name', 'obj_name', 'program_id',
               'instrument', 'date', 'mjd', 'exp_time',
               'obsclass', 'obstype', 'grating', 'filter1', 'filter2']

    data = list(
        zip(filename_list, obsid_list, objname_list, programid_list,
            instrument_list, date_list, mjd_list, exptime_list,
            obs_class_list, obs_type_list, grating_list, filter1_list,
            filter2_list))

    df = pd.DataFrame(data, columns=columns)

    df = df.sort_values(by=[sortby])

    if save and len(df) > 0:
        if save_name is not None:
            df.to_csv(datapath + save_name + '.csv', index=False)
            if verbosity > 0:
                print('[INFO] Save header information to csv file: {}'.format(
                    datapath + save_name + '.csv'))
        else:
            df.to_csv(datapath + 'fitslist.csv', index=False)
            if verbosity > 0:
                print('[INFO] Save header information to csv file: {}'.format(
                    datapath + 'fitslist.csv'))

    return df



def clean_nir_table(df, data_dir, delta_mjd=0.65, sortby='mjd',
                    calibrations='individual'):


    # Change the airmass to a numeric value to sort on
    df.airmass = pd.to_numeric(df.airmass, errors='coerce')

    # Create a column to indicate which frames were selected
    df.loc[:, 'selected'] = False

    # Select science images
    science_targets = df.query('frametype=="science" or '
                               'frametype=="arc,science,tilt"').copy()

    # Mark the selected science frames
    sel_idx = science_targets.index
    df.loc[sel_idx, 'selected'] = True

    # Sort science targets by mjd
    science_targets.sort_values(by=[sortby], inplace=True)
    science_targets.reset_index(drop=True, inplace=True)

    # -------------------------------------------------------
    # Science Images
    # -------------------------------------------------------

    # For the science targets check the offsets along the slit
    for index in science_targets.index:
        filename = data_dir + science_targets.loc[index, 'filename']
        hdr = fits.open(filename)[0].header
        offset_x_name = 'XOFFSET'
        if offset_x_name in hdr:
            offset_x = hdr[offset_x_name]
        else:
            offset_x = 0
        offset_y_name = 'YOFFSET'
        if offset_y_name in hdr:
            offset_y = hdr[offset_y_name]
        else:
            offset_x = 0
        science_targets.loc[index, 'slit_offset_x'] = offset_x
        science_targets.loc[index, 'slit_offset_y'] = offset_y

    num = 1

    # Resetting the comb_id and bkg_id values
    science_targets.loc[:, 'comb_id'] = None
    science_targets.loc[:, 'bkg_id'] = None

    for index in science_targets.index:
        combid = science_targets.loc[index, 'comb_id']
        bkgid = science_targets.loc[index, 'bkg_id']

        # Only populate comb_id and bkg_id, if they are empty
        if combid is None and bkgid is None:
            if index + 1 in science_targets.index:
                # Check if next telluric frame matches AB pattern
                name = science_targets.loc[index, 'target']
                next_name = science_targets.loc[index + 1, 'target']
                offset_diff = abs(science_targets.loc[index, 'slit_offset_x'] -
                                  science_targets.loc[
                                      index + 1, 'slit_offset_x'])

                if name == next_name and offset_diff > 2.5:
                    science_targets.loc[index, 'comb_id'] = int(num)
                    science_targets.loc[index + 1, 'comb_id'] = int(num + 1)
                    science_targets.loc[index, 'bkg_id'] = int(num + 1)
                    science_targets.loc[index + 1, 'bkg_id'] = int(num)
                    num += 2

                else:
                    science_targets.loc[index, 'comb_id'] = int(num)
                    science_targets.loc[index, 'bkg_id'] = -1

                    num += 1
            else:
                science_targets.loc[index, 'comb_id'] = int(num)
                science_targets.loc[index, 'bkg_id'] = -1

                num += 1

    # Change frametype to 'tilt, arc,science'
    for idx, index in enumerate(science_targets.index):
        if calibrations == 'individual':
            science_targets.loc[idx, 'calib'] = idx
        elif calibrations == 'same':
            science_targets.loc[idx, 'calib'] = 0

        science_targets.loc[index, 'frametype'] = 'arc,science,tilt'


    # -------------------------------------------------------
    # Tellurics
    # -------------------------------------------------------

    # Create groups by binning and slit from the science_targets
    groups = science_targets.groupby(['decker', 'binning'])

    # Create empty tellurics DataFrame
    tellurics = pd.DataFrame()

    for key in groups.indices.keys():

        tell = df.query('(frametype=="standard" or '
                        'frametype=="arc,standard,tilt") and '
                        'decker=="{}" and binning=="{}"'.format(key[0],
                                                                key[1])).copy()

        # For the tellurics read the offsets along the slit and
        # rename telluric target name according to header keyword
        for index in tell.index:
            filename = data_dir + tell.loc[index, 'filename']
            hdr = fits.open(filename)[0].header
            offset_x_name = 'XOFFSET'
            if offset_x_name in hdr:
                offset_x = hdr[offset_x_name]
            else:
                offset_x = 0
            offset_y_name = 'YOFFSET'
            if offset_y_name in hdr:
                offset_y = hdr[offset_y_name]
            else:
                offset_y = 0
            tell.loc[index, 'slit_offset_x'] = offset_x
            tell.loc[index, 'slit_offset_y'] = offset_y
            target_name = tell.loc[index, 'target']
            target_name = ''.join(target_name.split(' '))
            tell.loc[index, 'target'] = target_name + '_tell'

        # Mark the selected tellurics frames
        sel_idx = tell.index
        df.loc[sel_idx, 'selected'] = True

        # Set the calib values to the closest science image with same
        # decker and binning
        for idx in tell.index:
            sci = science_targets.copy()
            sci = sci.query('decker=="{}" and binning=="{}"'.format(key[0],
                                                                    key[
                                                                        1])).copy()

            # find the closest science target in mjd
            tell_mjd = float(tell.loc[idx, 'mjd'])
            sci.loc[:, 'mjd'] = pd.to_numeric(sci.loc[:, 'mjd'])
            sci.loc[:, 'mjd_diff'] = abs(sci.loc[:, 'mjd'] - tell_mjd)
            ydx = np.argmin(np.array(sci['mjd_diff']))
            tell.loc[idx, 'calib'] = sci.loc[sci.index[ydx], 'calib']

        tellurics = tellurics.append(tell)

    # Renumber combination and background IDs for the tellurics
    # The routine checks for AB pairs and treats single exposures correctly.

    # Resetting the comb_id and bkg_id values
    tellurics.loc[:, 'comb_id'] = None
    tellurics.loc[:, 'bkg_id'] = None
    # Sort tellurics by mjd and reset index
    # The assumption is that AB patterns will be consecutive in MJD
    tellurics.sort_values(by=[sortby], inplace=True)
    tellurics.reset_index(drop=True, inplace=True)

    for index in tellurics.index:
        combid = tellurics.loc[index, 'comb_id']
        bkgid = tellurics.loc[index, 'bkg_id']
        # Only populate comb_id and bkg_id, if they are empty
        if combid is None and bkgid is None:
            if index + 1 in tellurics.index:
                # Check if next telluric frame matches AB pattern
                name = tellurics.loc[index, 'target']
                next_name = tellurics.loc[index + 1, 'target']
                offset_diff = abs(tellurics.loc[index, 'slit_offset_x'] -
                                  tellurics.loc[index + 1, 'slit_offset_x'])

                if name == next_name and offset_diff > 2:
                    tellurics.loc[index, 'comb_id'] = int(num)
                    tellurics.loc[index + 1, 'comb_id'] = int(num + 1)
                    tellurics.loc[index, 'bkg_id'] = int(num + 1)
                    tellurics.loc[index + 1, 'bkg_id'] = int(num)
                    num += 2

                else:
                    tellurics.loc[index, 'comb_id'] = int(num)
                    tellurics.loc[index, 'bkg_id'] = -1

                    num += 1
            else:
                tellurics.loc[index, 'comb_id'] = int(num)
                tellurics.loc[index, 'bkg_id'] = -1

                num += 1

    # -------------------------------------------------------
    # Pixelflats
    # -------------------------------------------------------

    # Create empty pixelflat DataFrame
    pixelflats = pd.DataFrame()

    for key in groups.indices.keys():
        pflats = df.query('(frametype=="trace,pixelflat" or '
                          'frametype=="pixelflat,trace") and '
                          'target=="GCALflat" and decker=="{}" and '
                          'binning=="{}"'.format(key[0], key[1])).copy()


        # Mark the selected pixelflat frames
        sel_idx = pflats.index
        df.loc[sel_idx, 'selected'] = True

        # Populate the calib column of the pixelflats

        # Select science files with same decker and binning
        sci = science_targets.copy()
        sci = sci.query('decker=="{}" and binning=="{}"'.format(key[0],
                                                                key[1])).copy()

        # Loop through all pixelflats
        for idx in pflats.index:
            mjd = pflats.loc[idx, 'mjd']
            # Select all science image calib values within +-delta_mjd
            calib_ids = sci.query('{0:}-{1:} <= mjd <= {0:}+{1:}'.format(
                mjd, delta_mjd)).calib.value_counts().index

            if calibrations == 'individual':
                # Prepare a string with the calib values
                calib = ''
                for cal in calib_ids:
                    calib += str(int(cal)) + ','
                # Remove trailing comma for calib string
                calib = calib[:-1]
                # Add calib string to pixelflat
                pflats.loc[idx, 'calib'] = calib
            elif calibrations == 'same':
                pflats.loc[idx, 'calib'] = 0

        pixelflats = pixelflats.append(pflats)

    # Create a dataframe with all not selected entries
    not_selected = df.query('selected==False').copy()
    not_selected.drop(labels='selected', axis=1, inplace=True)

    # Consolidate all selected frames in one DataFrame
    if tellurics.shape[0] > 0:
        tellurics.drop(labels='slit_offset_x', axis=1, inplace=True)
        tellurics.drop(labels='slit_offset_y', axis=1, inplace=True)
    science_targets.drop(labels='slit_offset_x', axis=1, inplace=True)
    science_targets.drop(labels='slit_offset_y', axis=1, inplace=True)

    pypeit_input = pd.DataFrame()
    pypeit_input = pypeit_input.append(science_targets)

    if tellurics.shape[0] > 0:
        pypeit_input = pypeit_input.append(tellurics)
    pypeit_input = pypeit_input.append(pixelflats)
    pypeit_input.drop(labels='selected', axis=1, inplace=True)

    pypeit_input.sort_values(by=[sortby], inplace=True)
    pypeit_input.fillna("None", inplace=True)
    return pypeit_input, not_selected



def prepare_gnirs_data(path, remove_originals=False,
                       verbosity=0, mode=None, delta_mjd=0.65,
                       sortby='mjd', cleaned_data=True,
                       calibrations='individual'):

    # Step 1 preparing the raw data.

    if mode is None or mode == 'data':

        if verbosity > 0:
            print('[INFO] Preparing GNIRS data')

            if not os.path.exists(path + '/reduced'):
                if verbosity > 0:
                    print('[INFO] Creating /reduced directory')
                os.makedirs(path + '/reduced/')

            if not os.path.exists(path + '/raw'):
                if verbosity > 0:
                    print('[INFO] Creating /raw directory')
                os.makedirs(path + '/raw/')

        # remove all non-necessary files
        for file in glob.glob(path + '*.txt'):
            if verbosity > 1:
                print('[INFO] Removing unnecessary file {}'.format(file))
            os.remove(file)

        # Untaring all file *.tar files
        untar_files(path, verbosity, remove_originals)


        # Unzipping all *.bz2 files
        unzip_files_bz2(path, verbosity, remove_originals)


        # Create the fits header DataFrame
        fits_df = make_image_df_gnirs(path, save=True,
                                      save_name='fits_list',
                                      verbosity=verbosity)

        # Loop through fits list and delete or copy files to according folders
        if verbosity > 0:
            print('[INFO] Sorting and copying files into raw folder')
        if remove_originals:
            if verbosity > 0:
                print('[INFO] Followed by removing original fits files')
        for idx in fits_df.index:
            filename = fits_df.loc[idx, 'filename']
            fits_name = filename.split('/')[-1]

            shutil.copy2(filename, '{}/raw/{}'.format(path, fits_name))
            if remove_originals:
                os.remove(filename)

    if mode is None or mode == 'cleanir':
        if verbosity > 0:
            print('[INFO] Cleaning the fits files with cleanir.py')

        os.chdir('{}/raw'.format(path))
        for filename in glob.glob('N*.fits'):
            os.system('cleanir.py -fq --src=200:800,1:1024 "{}"'.format(
                filename))

    # Revert back to main path
    os.chdir(path)

    if mode is None or mode == 'setup':

        # Run basic pypeit setups inside reduced folders
        abs_raw_path = '{}/raw'.format(path)
        relraw_path = '../raw'

        if verbosity > 0:
            print('[INFO] Run the preliminary pypeit_setup')
            print('[INFO] pypeit_setup -s gemini_gnirs -r {}'
                  '/'.format(abs_raw_path))

        # Move into the reduced folder
        os.chdir('{}/reduced/'.format(path))
        os.system('pypeit_setup -s gemini_gnirs -r {}'
                  '/'.format(abs_raw_path))

        # Run full pypeit setups inside reduced folders
        if verbosity > 0:
            print('[INFO] Run the pypeit_setup')
            print('[INFO] pypeit_setup -s gemini_gnirs -r {}'
                  '/ -b -c=all'.format(abs_raw_path))

        # Back up directory if it already exists
        if os.path.isdir(path + '/reduced/gemini_gnirs_A'):
            orig_dir = path + '/reduced/gemini_gnirs_A'
            now = datetime.datetime.now()
            backup_dir = path + '/reduced/gemini_gnirs_A_backup_{}'.format(
                now.strftime("%Y-%m-%d_%H-%M"))
            copy_tree(orig_dir, backup_dir)

        os.system('pypeit_setup -s gemini_gnirs -r {}'
                  '/ -b -c=all'.format(abs_raw_path))

    # Revert back to main path
    os.chdir(path)

    if mode is None or mode == 'prepare':

        # Run the pypeit table cleaning algorithm for VIS

        if verbosity > 0:
            print('[INFO] Preparing a clean pypeit input table')
        setup_path = path + '/reduced' \
                           '/setup_files'
        raw_path = path + '/raw/'
        sorted_files = glob.glob(setup_path +
                                 '/gemini_gnirs_*.sorted')

        for file in sorted_files:
            if verbosity > 0:
                print('[INFO] Cleaning {}'.format(file))
            df = read_sorted_file(file)

            # Either use cleaned or uncleaned images
            print(df['filename'].str)
            if cleaned_data:
                df = df[df['filename'].str.startswith('c')].copy()
            else:
                df = df[df['filename'].str.startswith('N')].copy()

            df.to_csv(setup_path+'/gemini_gnirs_sorted.csv')

            # Now clean the NIR table
            cleaned_df, not_selected = clean_nir_table(df, raw_path,
                                                       delta_mjd=delta_mjd,
                                                       sortby=sortby,
                                                       calibrations=calibrations)

            cleaned_df.to_csv(setup_path+'/gemini_gnirs_prepared.csv')
            write_sorted_table(cleaned_df,
                               setup_path +
                               '/gemini_gnirs_suggested_table.txt',
                               sortby=sortby)
            if not_selected.shape[0] >0:
                write_sorted_table(not_selected,
                                   setup_path +
                                   '/gemini_gnirs_disregarded_table.txt',
                                   sortby=sortby)
