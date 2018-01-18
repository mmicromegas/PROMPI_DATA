import PROMPI_data as prd
import matplotlib.pyplot as plt

dataloc = ''  # folder with ransdat data, replace tseries_ransdat > store final averaging to a file, introduce check if final averagring done, 
filename_rans = 'ob3d.45.lrez.00041.ransdat' 
filename_blck = 'ob3d.45.lrez.00041.blockdat'

ob_rans = prd.PROMPI_rans(filename_rans)
ob_blck = prd.PROMPI_blck(filename_blck)

data = ob_rans.rans()
#print(data.shape,data.size)
#rl = ob_rans.ranslist()
#print(rl)
#print(data[2,2,:])
#print(type(data))
ob_rans.ransdict()
#print(data[0:12])
#print(data['eh_enuc2'])

plt.plot(data['xzn0'],data['eh_enuc1'])
plt.show()