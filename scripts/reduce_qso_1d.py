

import os
import glob

import pandas as pd
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table

def write_header(file):

    file.write('# Automatically generated input file produced by the '
               'reduce_qso_1d script.\n')
    file.write('# ATTENTION: Please use at your own risk!\n')
    file.write('# Further questions can be directed to '
               'schindler@strw.leidenuniv.nl \n')

def read_sorted_table(filename):
    table = ascii.read(filename, delimiter='|', format='fixed_width')
    df = table.to_pandas()

    return df

def read_pypeit_file(filename):
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


def write_pypeit_parameters(file, pars):
    # Write parameters
    for key in pars:

        if type(pars[key]) == dict:
            file.write('[{}]\n'.format(key))

            subpar = pars[key]

            for p in subpar:
                if type(subpar[p]) == dict:
                    file.write('  [[{}]]\n'.format(p))
                    ssubpar = subpar[p]

                    for pp in ssubpar:
                        if type(ssubpar[pp]) == dict:
                            file.write('    [[[{}]]]\n'.format(pp))

                            sssubpar = ssubpar[pp]
                            for ppp in sssubpar:
                                file.write('      {} = {} \n'.format(ppp,
                                                                     sssubpar[
                                                                         ppp]))

                        elif ssubpar[pp] is not None:

                            file.write('    {} = {} \n'.format(pp, ssubpar[pp]))
                elif subpar[p] is not None:
                    file.write('  {} = {} \n'.format(p, subpar[p]))
        elif pars[key] is not None:
            file.write('{} = {} \n'.format(key, pars[key]))


default_flux_parameters = {'extinct_correct': False}

def make_flux_infile(objname, objfiles, sensfuncfiles, flux_ids=None,
                     fluxpars=None):

    infile_flux = '{}.flux'.format(objname)
    file = open(infile_flux, 'w+')

    # Write header
    write_header(file)
    file.write('\n')

    # Write parameters
    file.write('# User-defined execution parameters\n')
    write_pypeit_parameters(file, fluxpars)


    file.write('\n')
    file.write('# Read in the flux\n')
    file.write('flux read\n')

    if len(sensfuncfiles) == 1:
        file.write('{} {}\n'.format(objfiles[0], sensfuncfiles[0]))
        for objfile in objfiles[1:]:
            file.write('{}\n'.format(objfile))

    elif len(sensfuncfiles) == len(objfiles):
        for idx, objfile in enumerate(objfiles):
            file.write('{} {}\n'.format(objfile, sensfuncfiles[idx]))

    elif flux_ids is not None:

        if max(flux_ids) == len(sensfuncfiles)-1:
            for idx, objfile in enumerate(objfiles):
                file.write('{} {}\n'.format(objfile, sensfuncfiles[flux_ids[idx]]))

        else:
            raise ValueError('[ERROR] The number of different flux_ids does '
                             'not correspond to the number of sensfuncfiles.')

    else:
        raise ValueError('[ERROR] The number of given sensfuncfiles is larger '
                         'than 1 but incompatible to the number of object '
                         'files to flux.')

    file.write('flux end\n')

    return infile_flux


def make_coadd1d_infile(objname, objfiles, objids=None, coadd1dpars=None,
                        suffix=None):

    if suffix is not None:
        infile_coadd1d = '{}_{}.coadd1d'.format(objname, suffix)
    else:
        infile_coadd1d = '{}.coadd1d'.format(objname)
    file = open(infile_coadd1d, 'w+')

    write_header(file)
    file.write('\n')

    # Write parameters
    file.write('# User-defined execution parameters\n')
    write_pypeit_parameters(file, coadd1dpars)


    file.write('\n')
    file.write('# Read in the coadd1d\n')
    file.write('coadd1d read\n')

    for idx, objfile in enumerate(objfiles):
        file.write('{} {}\n'.format(objfile, objids[idx]))

    file.write('coadd1d end\n')

    return infile_coadd1d


def make_telcorr_infile(objname, telcorrpars=None, suffix=None):

    if suffix is not None:
        infile_tell = '{}_{}.tell'.format(objname, suffix)
    else:
        infile_tell = '{}.tell'.format(objname)
    file = open(infile_tell, 'w+')

    write_header(file)
    file.write('\n')

    # Write parameters
    file.write('# User-defined execution parameters\n')
    write_pypeit_parameters(file, telcorrpars)

    return infile_tell


def retrieve_objids(spec1dtxts, spec2dfiles):

    objids = []

    for jdx, file in enumerate(spec1dtxts):
        df = read_pypeit_file(file)

        unique_trace_ids = pd.unique(df.loc[:, 'name'])

        if len(unique_trace_ids) > 1:
            print('[WARNING] Multiple traces found!')
            print('[INFO]  index   trace')
            for idx, trace in enumerate(unique_trace_ids):
                print('    {}    {}'.format(idx, trace))
            print('[INFO] Opening corresponding 2D image')
            os.system('pypeit_show_2dspec {}'.format(spec2dfiles[jdx]))
            print('[INFO] Type in the index number for the relevant science '
                  'trace:')
            print('[INFO]  index   trace')
            for idx, trace in enumerate(unique_trace_ids):
                print('    {}    {}'.format(idx, trace))
            id = int(input())
        else:
            id = 0
        objids.append(unique_trace_ids[id])

    return objids


def oned_reduction(sourcename, sensfuncfiles, flux_ids=None,
                   fluxpars=default_flux_parameters,
                   coadd1dpars=None,
                   telcorrpars=None):

    # Retrieve information
    path = os.getcwd()
    spec2dfiles = glob.glob('./Science/spec2d*.fits'.format(path))
    spec1dfiles = glob.glob('./Science/spec1d*.fits'.format(path))
    spec1dtxts = glob.glob('./Science/spec1d*.txt'.format(path))

    # Sort by filename
    spec2dfiles.sort()
    spec1dtxts.sort()
    spec1dfiles.sort()


    # PREPARE ALL PYPEIT FILES
    # prepare fluxing file
    infile_flux = make_flux_infile(sourcename, spec1dfiles, sensfuncfiles,
                       flux_ids=flux_ids,
                     fluxpars=fluxpars)


    # prepare coadd1d file
    objids = retrieve_objids(spec1dtxts, spec2dfiles)


    infile_coadd1d = make_coadd1d_infile(sourcename, spec1dfiles, objids=objids,
                        coadd1dpars=coadd1dpars,
                        suffix=None)


    # prepare tellruic correction file
    infile_tell = make_telcorr_infile(sourcename, telcorrpars, suffix=None)


    # Execute steps

    # a) execute fluxing
    print('[INFO] Executing fluxing')
    os.system('pypeit_flux_calib {}'.format(infile_flux))

    # b) execute coadd1d
    print('[INFO] Executing coadd1d')
    os.system('pypeit_coadd_1dspec {}'.format(infile_coadd1d))

    # c) execute telluric correction for coadded file
    print('[INFO] Executing telluric correction')
    coadd_file = coadd1dpars['coadd1d']['coaddfile']
    os.system('pypeit_tellfit {} -t={}'.format(coadd_file, infile_tell))


    print('[INFO] 1D reduction complete!')