

import os
import copy
import numpy as np

from pypeit.core import telluric
# from pypeit.core.flux_calib import apply_sensfunc
# from pypeit.core import save
from pypeit import coadd1d
from pypeit import msgs
import glob



def write_header(file):

    file.write('# pypeit-helpers automatically generated this input file \n')
    # file.write('# \n')







# 1 checking the science directory and all spectra




# 2 fluxing

# 2.a creating the sensitivity function

# 3 coadding

# 4 telluric correction