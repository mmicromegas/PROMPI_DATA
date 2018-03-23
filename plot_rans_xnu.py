import PROMPI_RANS_xnu as prx
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# CHOOSE DATA SOURCE 
eht_data = 'tseries_ransdat.npy' 
data_prefix = 'EHT_'

# CHOOSE CENTRAL TIME
intc = 3

# CHOOSE XNUCID
inuc = '0006'

# LIMIT GRID (get rid of boundary noise)
xbl = 3.72e8
xbr = 9.8e8

# INSTANTIATE 
RANSX = prx.PROMPI_xnu(eht_data,inuc,intc)

# plot X transport equation 

RANSX.plot_Xrho(xbl,xbr,inuc,data_prefix)
RANSX.plot_Xtransport_equation(xbl,xbr,inuc,data_prefix)

# plot X flux equation

RANSX.plot_Xflux(xbl,xbr,inuc,data_prefix)
#RANSX.plot_Xflux_equation(xbl,xbr,inuc)

# plot X variance equation

RANSX.plot_Xvariance(xbl,xbr,inuc,data_prefix)
RANSX.plot_Xvariance_equation(xbl,xbr,inuc,data_prefix)

# plot X diffusivity (Eulerian, Lagrangian)

#RANSX.plot_X_Ediffusivity(xbl,xbr,inuc)
#RANSX.plot_X_Ldiffusivity(xbl,xbr,inuc)

# AMSTERDAM


