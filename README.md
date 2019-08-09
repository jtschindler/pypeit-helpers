# xshooter_pypeit_setup
Useful python scripts to prepare the X-SHOOTER data (with raw calibrations) for reduction with PypeIt


## Disclaimers
While this code has been test for multiple XShooter data sets, it might still fail for your use case. Therefore, always make sure to check the output of the code, especially the 'suggestion_table.txt' before you blindly copy it to your pypeIT file and run it. For example, if the automatically given calibrations from the ESO archive do not provide a suitable arc, then the arc will be missing in the 'vlt_xshooter...suggested_table.txt'.

Sky standards ('STD,SKY') are currently ignored and will not be part of the 'vlt_xshooter...suggested_table.txt'.

In some cases tellurics are not taken in sequences of even numbers of frames. In these cases the comb_id and bkg_id values will be wrong!

If the '...suggested_table.txt' does not list calib values for your biases or pixelflats, you should consider to change the deltamjd variable (--deltamjd=[SOMENUMBER]). It is responsible to find calibration files close to your science/telluric observations.

## Purpose
The provided python scripts prepare a folder with XShooter data for a pypeIT run. The data can either be automatically (data with raw calibrations) or manually downloaded from the ESO archive. At this moment it is only designed to work on the XShooter VIS and NIR arm. UVB data will be automatically disregarded.

1) It creates a folder structure for the reduction process
2) It removes unnecessary files and unzips all other fits files
3) It sorts the VIS/NIR data into separate folders
4) It executes the pypeit_setup scripts to create the .sorted and .pypeit files
5) It reads in the .sorted files and creates a suggestion ('suggestion_table.txt') for the final table in the .pypeit files. All rows removed in this process are saved in a different file ('diregard_table.txt')

## Installation
1) Clone this directory from github ('git clone ...')
2) Set the python routines to be executable ('chmod +x prepare_xshooter.py, chmod +x sort_xshooter.py')
3) Add the path to the uberPypeitXshooter folder to your path in your .bashrc or .profile
4) Test if it can be executed from the command line. Type 'prepare_xshooter.py -h' to display the help message.

## Usage
For the most general use case, navigate to your folder with the XShooter data (either .fits or .fits.Z files). In your command line type

prepare_xshooter.py -o '[TARGETNAME]'

and replace [TARGETNAME] by the name of your target (e.g. prepare_xshooter.py -o 'J0421-2657'). The code will run on the default settings. For a list of all input parameters use the '-h' help option.

### Options and default settings:

datadir: In its default setting it will start from the directory that you execute it in. However, you can specify a certain directory with '-d'.

verbosity: The code will let you know what it is doing, while it runs. The output can be modified by setting the verbosity '-v'. To disable all output use '-v=0', to increase the detail of the default output set it to a number larger than 1.

remove: In default it will only copy the useful VIS and NIR data files without removing them. To remove them use '-r=True'

mode: At last you can run different parts of the code with the '-m' option. Consult the help tooltip for the different settings.

deltamjd: We use a value of 0.65 here to associate calibrations with their science files. You can easily tune this difference to your wishes with '--deltamjd=[SOMENUMBER]'
