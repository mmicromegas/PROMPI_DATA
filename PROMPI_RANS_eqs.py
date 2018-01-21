import PROMPI_data as prd
import numpy as np
import matplotlib.pyplot as plt

class PROMPI_eqs:

    def __init__(self,filename):
        pass
			

    def CorrectMean_ux(self,tt):

        dmdt = self.dt(self.eht_mm,tt) 
        vexp = -dmdt/(4.*np.pi*self.xzn0*self.xzn0*self.eht_dd[:,tt])

        self.fht_ux[:,tt] = vexp[:]
        self.eht_ux[:,tt] = self.fht_ux[:,tt] - self.eht_ax[:,tt]
		
