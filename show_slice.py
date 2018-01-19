import PROMPI_data as prd
import matplotlib.pyplot as plt
import numpy as np

dataloc = 'DATA/'
filename_blck = dataloc+'ob3d.45.lrez.00001.blockdat'

ob_blck = prd.PROMPI_blockdat(filename_blck)

