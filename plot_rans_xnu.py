import PROMPI_RANS_xnu as prx
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# CHOOSE DATA SOURCE 
eht_data = 'tseries_ransout.npy' 


# CHOOSE PREFIX for figure names in RESULTS
data_prefix = 'tseries_ransdat'+'_'

# CHOOSE GEOMETRY OF THE SIMULATION
# ig = 1 CARTESIAN x,y,z
# ig = 2 SPHERICAL r,theta,phi

ig = 1

# CHOOSE CENTRAL TIME INDEX
intc = 5

# CHOOSE XNUCID
inuc = '0005'

# IF LGRID = 1, LIMIT THE GRID (get rid of boundary noise)
# IF LGRID = 0, DO NOT LIMIT THE GRID
LGRID = 1

# LIMIT GRID 

# O-burn Meakin/Arnett 2007
#xbl = 3.72e8
#xbr = 9.8e8

# Ne
xbl = 3.52e8
xbr = 3.92e8

# INSTANTIATE 
RANSX = prx.PROMPI_xnu(eht_data,ig,inuc,intc,LGRID,xbl,xbr)

# plot X transport equation 

RANSX.plot_Xrho(data_prefix)
RANSX.plot_Xtransport_equation(data_prefix)

# plot X flux equation

RANSX.plot_Xflux(data_prefix)
RANSX.plot_Xflux_equation(data_prefix)

# plot X variance equation

RANSX.plot_Xvariance(data_prefix)
RANSX.plot_Xvariance_equation(data_prefix)

# plot X diffusivity (Eulerian)

RANSX.plot_X_Ediffusivity(data_prefix)

# AMSTERDAM


