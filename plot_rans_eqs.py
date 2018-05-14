import PROMPI_RANS_eqs as pre
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

ig = 2

# CHOOSE CENTRAL TIME INDEX
intc = 5


# IF LGRID = 1, LIMIT THE GRID (get rid of boundary noise)
# IF LGRID = 0, DO NOT LIMIT THE GRID
LGRID = 0

# LIMIT GRID
 
# O-burn Meakin/Arnett 2007
#xbl = 3.72e8
#xbr = 9.8e8

# Ne
xbl = 3.52e8
xbr = 3.92e8

# INSTANTIATE 
RANS = pre.PROMPI_eqs(eht_data,ig,intc,LGRID,xbl,xbr)

# PLOT CONTINUITY EQUATION 

RANS.plot_rho(data_prefix)
RANS.plot_continuity_equation(data_prefix)
RANS.plot_continuity_equation_bar(data_prefix)

# PLOT UX MOMENTUM EQUATION

RANS.plot_ux(data_prefix)
RANS.plot_Rmomentum_equation(data_prefix)

# PLOT TURBULENT KINETIC ENERGY EQUATION

RANS.plot_tke(data_prefix)
RANS.plot_tke_equation(data_prefix)

# PLOT INTERNAL ENERGY EQUATION

RANS.plot_ei(data_prefix)
RANS.plot_ei_equation(data_prefix)

# PLOT ENTROPY EQUATION

RANS.plot_ss(data_prefix)
RANS.plot_ss_equation(data_prefix)

# PRINT PROPERTIES OF YOUR SIMULATION
RANS.properties()
