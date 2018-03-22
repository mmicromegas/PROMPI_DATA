import PROMPI_RANS_xnu as prx
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# CHOOSE DATA SOURCE
eht_data = 'EHT.npy' 

# CHOOSE CENTRAL TIME
intc = 1

# CHOOSE XNUCID
inuc = '0005'

# LIMIT GRID
xbl = 3.72e8
xbr = 9.8e8

# INSTANTIATE 
RANSX = prx.PROMPI_xnu(eht_data,inuc,intc)

# plot X transport equation 

RANSX.plot_Xrho(xbl,xbr,inuc)
#RANSX.plot_Xtransport_equation(xbl,xbr,inuc)

# plot X flux equation

#RANSX.plot_Xflux(xbl,xbr,inuc)
#RANSX.plot_Xflux_equation(xbl,xbr,inuc)

# plot X variance equation

#RANSX.plot_Xvariance(xbl,xbr,inuc)
#RANSX.plot_Xvariance_equation(xbl,xbr,inuc)

# plot X diffusivity (Eulerian, Lagrangian)

#RANSX.plot_X_Ediffusivity(xbl,xbr,inuc)
#RANSX.plot_X_Ldiffusivity(xbl,xbr,inuc)

# AMSTERDAM


