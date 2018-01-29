import PROMPI_data as prd
import matplotlib.pyplot as plt
import numpy as np

dataloc = 'DATA/'
filename_blck = dataloc+'ob3d.45.lrez.00001.blockdat'
#filename_blck = dataloc+'ob3d.45.mrez.01264.bindata'
dat = 'temp'

ob_blck = prd.PROMPI_blockdat(filename_blck,dat)

#print(ob_blck.dt())
#plt.plot(ob_blck.dt())
#plt.show()


#print(ob_blck.test().shape)
#help(ob_blck.test())

plt.plot(ob_blck.test())

#plt.plot(ob_blck.test()[:,100,100])
#plt.plot(ob_blck.test()[:,200,200])
#plt.plot(ob_blck.test()[:,150,200])

plt.show(block=False)
