#!/usr/bin/env python3

import os
import glob
import numpy as np

import shutil
import pandas as pd
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table

template_signature_to_delete = ['XSHOOTER_slt_acq',
                                'XSHOOTER_slt_cal_NIRArcsMultiplePinhole',
                                'XSHOOTER_slt_cal_VISLampFlatSinglePinhole',
                                'XSHOOTER_slt_cal_UVBVisArcsSinglePinhole',
                                'XSHOOTER_slt_cal_UVBVisArcsMultiplePinhole',
                                'XSHOOTER_slt_cal_NIRLampFlatSinglePinhole',
                                'XSHOOTER_slt_cal_NIRArcsSinglePinhole',
                                'XSHOOTER_slt_cal_NIRLampFlatSinglePinhole']

obj_name_to_delete = ['LAMP,FMTCHK', 'LAMP,ORDERDEF', 'LAMP,AFC']

def read_sorted_file(filename):

    with open(filename) as f:
        content = f.readlines()

    output = open('temp.txt', 'w+')
    for line in content:
        if '|' in line:
            output.write(line)
    output.close()

    df = read_sorted_table('temp.txt')

    # os.remove('temp.txt')
    return df

def read_sorted_table(filename):

    table = ascii.read(filename, delimiter='|', format='fixed_width')
    df = table.to_pandas()

    return df

def write_sorted_table(df, filename):

    table = Table.from_pandas(df)

    ascii.write(table, filename, delimiter='|', format='fixed_width',
                overwrite=True)

def make_image_df_xshooter(datapath, save=False, save_name=None, verbosity=0):
    """

    :param datapath:
    :param save:
    :param save_name:
    :return:
    """
    filename_list = []
    obs_name_list = []
    objname_list = []
    programid_list = []
    pi_list = []
    instrument_list = []
    date_list = []
    mjd_list = []
    exptime_list = []
    ra_list = []
    dec_list = []
    tpl_id_list = []
    dpr_catg_list = []
    dpr_tech_list = []
    dpr_type_list = []
    binx_list = []
    biny_list = []
    readout_clock_list = []
    origfname_list = []

    delete_list = []
    arm_list = []

    fits_list = glob.glob('{}/*.fits'.format(datapath))

    for file in fits_list:

        if verbosity > 2:
            print('[INFO] Reading fits header of {}'.format(file))

        hdu = fits.open(file)
        hdr = hdu[0].header

        filename_list.append(file)
        obs_name_list.append(hdr['HIERARCH ESO OBS NAME']) # Calibration' / OB
        # name
        objname_list.append(hdr['OBJECT']) # Original target
        programid_list.append(hdr['HIERARCH ESO OBS PROG ID']) # ESO program ID
        pi_list.append(hdr['PI-COI'])
        instrument_list.append(hdr['INSTRUME'])
        date_list.append(hdr['DATE-OBS'])
        mjd_list.append(hdr['MJD-OBS'])
        exptime_list.append(hdr['EXPTIME'])
        # ra_list.append(hdr['RA'])
        # dec_list.append(hdr['DEC'])
        tpl_id_list.append(hdr['HIERARCH ESO TPL ID']) # Template signature
        dpr_catg_list.append(hdr['HIERARCH ESO DPR CATG']) # Observation category
        dpr_tech_list.append(hdr['HIERARCH ESO DPR TECH']) #  Observation
        # technique
        dpr_type_list.append(hdr['HIERARCH ESO DPR TYPE']) # Observation type

        # binx_list.append(hdr['HIERARCH ESO DET WIN1 BINX'])  #Binning factor
        # along X
        # biny_list.append(hdr['HIERARCH ESO DET WIN1 BINY']) #B inning factor
        # along X
        # readout_clock_list.append(hdr['HIERARCH ESO DET READ CLOCK']) #
        # Readout clock pattern used
        origfname_list.append(hdr['ORIGFILE'])  # Original File Name


        # Populate the "delete" column
        delete = False
        if hdr['HIERARCH ESO TPL ID'] in template_signature_to_delete or hdr[
            'HIERARCH ESO OBS NAME'] in obj_name_to_delete:
            delete = True
        delete_list.append(delete)

        # Populate the "arm" column
        if 'NIR' in hdr['ORIGFILE']:
            arm_list.append('NIR')
        elif 'VIS' in hdr['ORIGFILE']:
            arm_list.append('VIS')
        else:
            arm_list.append('INDEF')


    # columns = ['filename', 'obs_name', 'obj_name', 'program_id', 'pi_coi',
    #            'instrument', 'date', 'mjd', 'exp_time', 'ra', 'dec',
    #            'template_signature', 'obs_category', 'obs_technique',
    #            'obs_type', 'bin_x', 'bin_y', 'readout_clock_pattern',
    #            'original_filename']
    #
    # data = list(
    #     zip(filename_list, obs_name_list, objname_list, programid_list,
    #         pi_list, instrument_list, date_list, mjd_list, exptime_list,
    #         ra_list, dec_list, tpl_id_list, dpr_catg_list, dpr_tech_list,
    #         dpr_type_list, binx_list, biny_list, readout_clock_list,
    #         origfname_list))

    columns = ['filename', 'obs_name', 'obj_name', 'program_id', 'pi_coi',
               'instrument', 'date', 'mjd', 'exp_time',
               'template_signature', 'obs_category', 'obs_technique',
               'obs_type',
               'original_filename', 'arm', 'deleted']

    data = list(
        zip(filename_list, obs_name_list, objname_list, programid_list,
            pi_list, instrument_list, date_list, mjd_list, exptime_list,
            tpl_id_list, dpr_catg_list, dpr_tech_list,
            dpr_type_list,
            origfname_list, arm_list, delete_list))

    df = pd.DataFrame(data, columns=columns)

    df = df.sort_values('mjd')

    if save and len(df) > 0:
        if save_name is not None:
            df.to_csv(datapath+save_name+'.csv', index=False)
            if verbosity >0:
                print('[INFO] Save header information to csv file: {}'.format(
                    datapath+save_name+'.csv'))
        else:
            df.to_csv(datapath+'fitslist.csv', index=False)
            if verbosity >0:
                print('[INFO] Save header information to csv file: {}'.format(
                    datapath+'fitslist.csv'))

    return df



def clean_nir_table(df, data_dir, delta_mjd=0.65):


    # Change the airmass to a numeric value to sort on
    df.airmass = pd.to_numeric(df.airmass, errors='coerce')

    # Create a column to indicate which frames were selected
    df.loc[:, 'selected'] = False

    # Select science images
    science_targets = df.query('frametype=="science" and target!="STD,'
                               'TELLURIC"').copy()

    # TAKE CARE of flux standards
    # Add flux standards to science list
    # flux_standards = df.query('frametype=="standard" and target=="STD,FLUX"')
    # science_targets = science_targets.append(flux_standards)

    # Mark the selected science frames
    sel_idx = science_targets.index
    df.loc[sel_idx, 'selected'] = True

    science_targets.sort_values('mjd', inplace=True)
    science_targets.reset_index(drop=True, inplace=True)

    # -------------------------------------------------------
    # Science Images
    # -------------------------------------------------------

    # Renumber combination and background IDs
    num = 1
    for idx, index in enumerate(science_targets.index):
        science_targets.loc[index,'comb_id'] = int(num)
        if num % 2 == 0:
            science_targets.loc[index, 'bkg_id'] = int(num - 1)
        else:
            science_targets.loc[index, 'bkg_id'] = int(num + 1)
        science_targets.loc[index, 'calib'] = int(num)
        num += 1

    # Change frametype to 'tilt, arc,science'
    for idx, index in enumerate(science_targets.index):
        science_targets.loc[idx, 'calib'] = idx
        science_targets.loc[index, 'frametype'] = 'tilt,arc,science'

    # TODO
    # Change frametype for flux standards to 'science'
    # Link flux standards with science observations
    # rename flux standard target column

    # -------------------------------------------------------
    # Tellurics
    # -------------------------------------------------------

    # Create groups by binning and slit from the science_targets
    groups = science_targets.groupby(['decker', 'binning'])

    # Create empty tellurics DataFrame
    tellurics = pd.DataFrame()

    for key in groups.indices.keys():

        tell = df.query('frametype=="science" and target=="STD,TELLURIC" and '
                        'decker=="{}" and binning=="{}"'.format(key[0],
                                                            key[1])).copy()

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
    for index in tellurics.index:
        tellurics.loc[index, 'comb_id'] = int(num)
        if num % 2 == 0:
            tellurics.loc[index, 'bkg_id'] = int(num - 1)
        else:
            tellurics.loc[index, 'bkg_id'] = int(num + 1)

        num += 1

    # Rename telluric target name according to header keyword
    for index in tellurics.index:
        filename = data_dir + tellurics.loc[index, 'filename']
        hdr = fits.open(filename)[0].header
        target_name = hdr['HIERARCH ESO OBS NAME']
        tellurics.loc[index, 'target'] = target_name+'_tell'

    # Change first frametype of tellurics to standard
    for index in tellurics.index:
        tellurics.loc[index, 'frametype'] = 'standard'

    # -------------------------------------------------------
    # Pixelflats
    # -------------------------------------------------------

    # Create empty tellurics DataFrame
    pixelflats = pd.DataFrame()

    for key in groups.indices.keys():
        pflats = df.query('(frametype=="trace,pixelflat" or '
                          'frametype=="pixelflat,trace") and target=="LAMP,'
                          'FLAT" and '
                        'decker=="{}" and binning=="{}"'.format(key[0],
                                                                key[1])).copy()
        # Only use every second pixelflat, as the other pixel flat are with
        # the lamps off, to limit persistence.
        # pflats = pflats[::2].copy()

        # Select the pixelflats with the highest exposure time
        pflats_exp_list = list(pflats.exptime.value_counts().index)
        pflats.query('exptime=={}'.format(max(pflats_exp_list)), inplace=True)

        # Mark the selected tellurics frames
        sel_idx = pflats.index
        df.loc[sel_idx, 'selected'] = True

        # Populate the calib column of the pixelflats

        # Select science files with same decker and binning
        sci = science_targets.copy()
        sci = sci.query('decker=="{}" and binning=="{}"'.format(key[0],
                                                                key[1])).copy()

        # Loop through all pixelflats
        for idx in pflats.index:
            mjd = pflats.loc[idx,'mjd']
            # Select all science image calib values within +-0.5 mjd
            calib_ids = sci.query('{0:}-{1:} <= mjd <= {0:}+{1:}'.format(
                mjd, delta_mjd)).calib.value_counts().index
            # Prepare a string with the calib values
            calib = ''
            for cal in calib_ids:
                calib += str(cal)+','
            # Remove trailing comma for calib string
            calib = calib[:-1]
            # Add calib string to pixelflat
            pflats.loc[idx,'calib'] = calib

        pixelflats = pixelflats.append(pflats)


    # Create a dataframe with all not selected entries
    not_selected = df.query('selected==False').copy()
    not_selected.drop(labels='selected', axis=1, inplace=True)


    # Consolidate all selected frames in one DataFrame
    pypeit_input = pd.DataFrame()
    pypeit_input = pypeit_input.append(science_targets)
    pypeit_input = pypeit_input.append(tellurics)
    pypeit_input = pypeit_input.append(pixelflats)
    pypeit_input.drop(labels='selected', axis=1, inplace=True)

    pypeit_input.sort_values('mjd', inplace=True)
    pypeit_input.fillna("None", inplace=True)


    return pypeit_input, not_selected


def clean_vis_table(df, data_dir, delta_mjd=0.65):

    # Change the airmass to a numeric value to sort on
    df.airmass = pd.to_numeric(df.airmass, errors='coerce')

    # Create a column to indicate which frames were selected
    df.loc[:, 'selected'] = False

    # Select science images
    science_targets = df.query('frametype=="science" and target!="STD,'
                               'TELLURIC"').copy()

    # TAKE CARE of flux standards
    # Add flux standards to science list
    # flux_standards = df.query('frametype=="standard" and target=="STD,FLUX"')
    # science_targets = science_targets.append(flux_standards)

    # Mark the selected science frames
    sel_idx = science_targets.index
    df.loc[sel_idx, 'selected'] = True

    science_targets.sort_values(by='mjd', inplace=True)
    science_targets.reset_index(drop=True, inplace=True)

    # -------------------------------------------------------
    # Science Images
    # -------------------------------------------------------

    # Modify, comb_id, bkg_id, calib and frametype
    num = 1
    for idx, index in enumerate(science_targets.index):
        science_targets.loc[index, 'comb_id'] = int(num)
        science_targets.loc[index, 'bkg_id'] = -1
        science_targets.loc[idx, 'calib'] = idx
        science_targets.loc[index, 'frametype'] = 'science'
        num += 1


    # TODO
    # Change frametype for flux standards to 'science'
    # Link flux standards with science observations
    # rename flux standard target column


    # -------------------------------------------------------
    # Tellurics
    # -------------------------------------------------------

    # Create groups by binning and slit from the science_targets
    groups = science_targets.groupby(['decker', 'binning'])

    # Create empty tellurics DataFrame
    tellurics = pd.DataFrame()

    for key in groups.indices.keys():

        tell = df.query('frametype=="science" and target=="STD,TELLURIC" and '
                        'decker=="{}" and binning=="{}"'.format(key[0],
                                                                key[1])).copy()

        # Mark the selected tellurics frames
        sel_idx = tell.index
        df.loc[sel_idx, 'selected'] = True

        # Set the calib values to the closest science image with same
        # decker and binning
        for idx in tell.index:
            sci = science_targets.copy()
            sci = sci.query(
                'decker=="{}" and binning=="{}"'.format(key[0], key[1])).copy()

            # find the closest science target in mjd
            tell_mjd = float(tell.loc[idx, 'mjd'])
            sci.loc[:, 'mjd'] = pd.to_numeric(sci.loc[:, 'mjd'])
            sci.loc[:, 'mjd_diff'] = abs(sci.loc[:, 'mjd'] - tell_mjd)
            ydx = np.argmin(np.array(sci['mjd_diff']))
            tell.loc[idx, 'calib'] = sci.loc[sci.index[ydx], 'calib']

        tellurics = tellurics.append(tell)

    # Renumber combination and background IDs for the tellurics
    for index in tellurics.index:
        tellurics.loc[index, 'comb_id'] = int(num)
        tellurics.loc[index, 'bkg_id'] = -1
        num += 1

        filename = data_dir + tellurics.loc[index, 'filename']
        hdr = fits.open(filename)[0].header
        target_name = hdr['HIERARCH ESO OBS NAME']
        tellurics.loc[index, 'target'] = target_name + '_tell'

    # Change first frametype of tellurics to standard
    for index in tellurics.index:
        tellurics.loc[index, 'frametype'] = 'standard'

    # -------------------------------------------------------
    # Biases
    # -------------------------------------------------------

    # Create empty bias DataFrame
    biases = pd.DataFrame()

    # Select biases with the same binning as science
    for key in groups.indices.keys():
        bs = df.query('frametype=="bias" and target=="BIAS" and '
                          'binning=="{}"'.format(key[1])).copy()

        # Mark the selected tellurics frames
        sel_idx = bs.index
        df.loc[sel_idx, 'selected'] = True

        # Select science files with same binning
        sci = science_targets.copy()
        sci = sci.query('binning=="{}"'.format(key[1]))

        # Loop through all biases to populate the calib values
        for idx in bs.index:
            mjd = bs.loc[idx, 'mjd']
            # Select all science image calib values within +-0.65 mjd
            calib_ids = sci.query('{0:}-{1:} <= mjd <= {0:}+{1:}'.format(
                mjd, delta_mjd)).calib.value_counts().index
            # Prepare a string with the calib values
            calib = ''
            for cal in calib_ids:
                calib += str(cal) + ','
            # Remove trailing comma for calib string
            calib = calib[:-1]
            # Add calib string to pixelflat
            bs.loc[idx, 'calib'] = calib

        biases = biases.append(bs)

    # -------------------------------------------------------
    # Pixelflats
    # -------------------------------------------------------

    # Create empty pixelflat DataFrame
    pixelflats = pd.DataFrame()

    for key in groups.indices.keys():
        pflats = df.query('(frametype=="pixelflat,trace" or '
                          'frametype=="trace,pixelflat") and target=="LAMP,'
                          'FLAT" and '
                          'decker=="{}" and binning=="{}"'.format(key[0],
                                                                  key[1])).copy()
        # Select the pixelflats with the highest exposure time
        pflats_exp_list = list(pflats.exptime.value_counts().index)
        pflats.query('exptime=={}'.format(max(pflats_exp_list)), inplace=True)

        # Mark the selected tellurics frames
        sel_idx = pflats.index
        df.loc[sel_idx, 'selected'] = True

        # Populate the calib column of the pixelflats

        # Select science files with same decker and binning
        sci = science_targets.copy()
        sci = sci.query('decker=="{}" and binning=="{}"'.format(key[0], key[1]))

        # Loop through all pixelflats
        for idx in pflats.index:
            mjd = pflats.loc[idx, 'mjd']
            # Select all science image calib values within +-0.5 mjd
            calib_ids = sci.query('{0:}-{1:} <= mjd <= {0:}+{1:}'.format(
                mjd, delta_mjd)).calib.value_counts().index
            # Prepare a string with the calib values
            calib = ''
            for cal in calib_ids:
                calib += str(cal) + ','
            # Remove trailing comma for calib string
            calib = calib[:-1]
            # Add calib string to pixelflat
            pflats.loc[idx, 'calib'] = calib

        pixelflats = pixelflats.append(pflats)

    # -------------------------------------------------------
    # Arcs
    # -------------------------------------------------------

    # # Create groups by binning and slit from the science_targets
    groups = science_targets.groupby(['decker', 'binning'])

    # Create empty tellurics DataFrame
    arcs = pd.DataFrame()

    for key in groups.indices.keys():
        acs = df.query('(frametype=="arc,tilt" or frametype=="tilt,arc") and '
                       'target=="LAMP,WAVE" and '
                       'decker=="{}" and binning=="{}"'.format(key[0],
                                                                  key[1])).copy()
        if key[1] == '2,2':
            acs = acs.append(df.query('(frametype=="arc,tilt" or '
                                      'frametype=="tilt,arc") and '
                                      'target=="LAMP,WAVE" and '
                       'decker=="{}" and binning=="{}"'.format(key[0],
                                                                  '1,1')).copy())

        # Select science files with same decker and binning
        sci = science_targets.copy()
        sci = sci.query(
            'decker=="{}" and binning=="{}"'.format(key[0], key[1]))

        # Loop through all pixelflats
        for idx in acs.index:
            mjd = acs.loc[idx, 'mjd']
            # Select all science image calib values within +-0.5 mjd
            calib_ids = sci.query('{0:}-{1:} <= mjd <= {0:}+{1:}'.format(
                mjd, delta_mjd)).calib.value_counts().index
            # Prepare a string with the calib values
            calib = ''
            for cal in calib_ids:
                calib += str(cal) + ','
            # Remove trailing comma for calib string
            calib = calib[:-1]
            # Add calib string to pixelflat
            acs.loc[idx, 'calib'] = calib

        arcs = arcs.append(acs)

    # Create a dataframe with all not selected entries
    not_selected = df.query('selected==False').copy()
    not_selected.drop(labels='selected', axis=1, inplace=True)

    # Consolidate all selected frames in one DataFrame
    pypeit_input = pd.DataFrame()
    pypeit_input = pypeit_input.append(science_targets)
    pypeit_input = pypeit_input.append(tellurics)
    pypeit_input = pypeit_input.append(pixelflats)
    pypeit_input = pypeit_input.append(biases)
    pypeit_input = pypeit_input.append(arcs)
    pypeit_input.drop(labels='selected', axis=1, inplace=True)

    pypeit_input.sort_values('mjd', inplace=True)
    pypeit_input.fillna("None", inplace=True)

    return pypeit_input, not_selected


def prepare_xshooter_data(path, obj_name, remove_originals=False,
                          verbosity=0, mode=None, delta_mjd=0.65):

    if mode is None or mode == 'data':

        if verbosity > 0:
            print('[INFO] Preparing XShooter data')

        if not os.path.exists(path+'/raw/{}/NIR/'.format(obj_name)):
            if verbosity > 0:
                print('[INFO] Creating /raw/{}/NIR directory'.format(obj_name))
            os.makedirs(path+'/raw/{}/NIR/'.format(obj_name))

        if not os.path.exists(path+'/raw/{}/VIS/'.format(obj_name)):
            if verbosity > 0:
                print('[INFO] Creating /raw/{}/VIS directory'.format(obj_name))
            os.makedirs(path+'/raw/{}/VIS/'.format(obj_name))

        if not os.path.exists(path+'/reduced/{}/NIR/'.format(obj_name)):
            if verbosity > 0:
                print('[INFO] Creating /reduced/{}/NIR directory'.format(obj_name))
            os.makedirs(path+'/reduced/{}/NIR/'.format(obj_name))

        if not os.path.exists(path+'/reduced/{}/VIS/'.format(obj_name)):
            if verbosity > 0:
                print('[INFO] Creating /reduced/{}/VIS directory'.format(obj_name))
            os.makedirs(path+'/reduced/{}/VIS/'.format(obj_name))


        # remove all non-necessary files
        if verbosity > 0:
            print('[INFO] Removing unnecessary files')
        for file in glob.glob(path+'M.*.fits'):
            if verbosity > 1:
                print('[INFO] Removing unnecessary file {}'.format(file))
            os.remove(file)
        for file in glob.glob(path+'*.xml'):
            if verbosity > 1:
                print('[INFO] Removing unnecessary file {}'.format(file))
            os.remove(file)
        for file in glob.glob(path+'*.NL.txt'):
            if verbosity > 1:
                print('[INFO] Removing unnecessary file {}'.format(file))
            os.remove(file)

        # Gunzipping all *.fits.Z files
        files_to_unzip = glob.glob(path + '*.fits.Z')
        if verbosity > 0:
            print('[INFO] Gunzipping all *.fits.Z files in path')
        for file in files_to_unzip:
            if verbosity > 2:
                print('[INFO] Gunzipping {}'.format(file))
            os.system('gunzip {}'.format(file))

        # Creating the fits header DataFrame
        fits_df = make_image_df_xshooter(path, save=True,
                               save_name='fits_list', verbosity=verbosity)


        # Loop through fits list and delete or copy files to according folders
        if verbosity > 0:
            print('[INFO] Sorting and copying files into NIR/VIS folders')
        if remove_originals:
            if verbosity > 0:
                print('[INFO] And removing original fits files')
        for idx in fits_df.index:
            filename = fits_df.loc[idx, 'filename']
            fits_name = filename.split('/')[-1]

            if fits_df.loc[idx, 'deleted']:
                os.remove(filename)
            else:
                if fits_df.loc[idx, 'arm'] == 'NIR':
                    shutil.copy2(filename, '{}/raw/{}/{}/{}'.format(path, obj_name,
                                                                    'NIR',
                                                                    fits_name))
                    if remove_originals:
                        os.remove(filename)
                elif fits_df.loc[idx, 'arm'] == 'VIS':
                    shutil.copy2(filename, '{}/raw/{}/{}/{}'.format(path,
                                                                    obj_name,
                                                                    'VIS',
                                                                    fits_name))
                    if remove_originals:
                        os.remove(filename)



    if mode is None or mode == 'setup':
        # Run preliminary pypeit setups inside reduced folders
        print(path)
        os.chdir(path)
        cwd = os.getcwd()
        rawobj_path = cwd+'/raw/{}'.format(obj_name)
        relraw_path = '../../../raw/{}'.format(obj_name)

        if verbosity > 0:
            print('[INFO] Run the preliminary VIS pypeit_setup')
            print('[INFO] pypeit_setup -s vlt_xshooter_vis -r {}/VIS/'.format(relraw_path))
        os.chdir(cwd+'/reduced/{}/VIS/'.format(obj_name))
        os.system('pypeit_setup -s vlt_xshooter_vis -r {}/VIS/'.format(relraw_path))

        if verbosity > 0:
            print('[INFO] Run the preliminary NIR pypeit_setup')
            print('[INFO] pypeit_setup -s vlt_xshooter_nir -r {}/NIR/'.format(relraw_path))
        os.chdir(cwd + '/reduced/{}/NIR/'.format(obj_name))
        os.system('pypeit_setup -s vlt_xshooter_nir -r {}/NIR/'.format(relraw_path))



        # Run full pypeit setups inside reduced folders
        os.chdir(path)
        cwd = os.getcwd()
        rawobj_path = cwd+'/raw/{}'.format(obj_name)
        relraw_path = '../../../raw/{}'.format(obj_name)

        if verbosity > 0:
            print('[INFO] Run the VIS pypeit_setup')
            print('[INFO] pypeit_setup -s vlt_xshooter_vis -r {}/VIS/ -b '
                  '-c=all'.format(relraw_path))
        os.chdir(cwd+'/reduced/{}/VIS/'.format(obj_name))
        os.system('pypeit_setup -s vlt_xshooter_vis -r {}/VIS/ -b -c=all'.format(
            relraw_path))

        if verbosity > 0:
            print('[INFO] Run the NIR pypeit_setup')
            print('[INFO] pypeit_setup -s vlt_xshooter_nir -r {}/NIR/ -b '
                  '-c=all'.format(relraw_path))
        os.chdir(cwd + '/reduced/{}/NIR/'.format(obj_name))
        os.system('pypeit_setup -s vlt_xshooter_nir -r {}/NIR/ -b -c=all'.format(
            relraw_path))

    if mode is None or mode == 'clean':
        os.chdir(path)
        cwd = os.getcwd()
        # Run the pypeit table cleaning algorithm for VIS
        if verbosity > 0:
            print('[INFO] Preparing a clean pypeit input table for VIS')
        vis_setup_path = cwd + '/reduced/{}/VIS/setup_files'.format(obj_name)
        vis_raw_path = cwd + '/raw/{}/VIS/'.format(obj_name)
        vis_sorted_files = glob.glob(vis_setup_path + '/vlt_xshooter_vis_*.sorted')


        for vis_file in vis_sorted_files:
            if verbosity > 0:
                print('[INFO] Cleaning {}'.format(vis_file))
            vis_file_name = vis_file.split('.')[0]
            df = read_sorted_file(vis_file)
            df.to_csv('{}.csv'.format(vis_file_name))
            cleaned_df, not_selected = clean_vis_table(df, vis_raw_path,
                                                       delta_mjd=delta_mjd)
            cleaned_df.to_csv('{}_cleaned.csv'.format(vis_file_name))
            write_sorted_table(cleaned_df,'{}_suggested_table.txt'.format(vis_file_name))
            write_sorted_table(not_selected, '{}_disregarded_table.txt'.format(vis_file_name))


        # Run the pypeit table cleaning algorithm for NIR
        if verbosity > 0:
            print('[INFO] Preparing a clean pypeit input table for NIR')
        nir_setup_path = cwd+'/reduced/{}/NIR/setup_files'.format(obj_name)
        nir_raw_path = cwd+'/raw/{}/NIR/'.format(obj_name)
        nir_sorted_files = glob.glob(nir_setup_path+'/vlt_xshooter_nir_*.sorted')


        for nir_file in nir_sorted_files:
            if verbosity > 0:
                print('[INFO] Cleaning {}'.format(nir_file))
            nir_file_name = nir_file.split('.')[0]
            df = read_sorted_file(nir_file)
            df.to_csv('{}.csv'.format(nir_file_name))
            cleaned_df, not_selected = clean_nir_table(df, nir_raw_path,
                                                       delta_mjd=delta_mjd)
            cleaned_df.to_csv('{}_cleaned.csv'.format(nir_file_name))
            write_sorted_table(cleaned_df, '{}_suggested_table.txt'.format(
                                   nir_file_name))
            write_sorted_table(not_selected, '{}_disregarded_table.txt'.format(
                nir_file_name))




