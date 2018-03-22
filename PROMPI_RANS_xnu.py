import PROMPI_data as prd
import numpy as np
import matplotlib.pyplot as plt
import CALCULUS as calc

class PROMPI_xnu:

    def __init__(self,filename,inuc,intc):
	
        # load data to structured array
        eht = np.load(filename)	

        self.xzn0        = eht.item().get('rr') 
        self.eht_dd      = eht.item().get('dd')[intc]
        self.eht_ddxi    = eht.item().get('ddx'+inuc)[intc]
#        self.eht_ddux    = eht['ddux'][intc]
#        self.eht_ddxiux  = eht['ddx'+inuc+'ux']
#        self.eht_ddxidot = eht['ddx'+inuc+'dot']	
	
    def plot_Xrho(self,xbl,xbr,inuc):
        """Plot Xrho stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(inuc)
		
        # load x GRID
        grd1 = self.xzn0
		
		# load DATA to plot
        plt1 = self.eht_ddxi
		
		# calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
		# initiate FIGURE
        fig, ax1 = plt.subplots(figsize=(7,6))
		
		# limit x/y axis
        ax1.axis([xbl,xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
		
		# plot DATA 
        # ax1.title(xnucid)
        ax1.plot(grd1,plt1,color='b',label = r'$\overline{\rho} \widetilde{X}$')

        # define and show x/y LABELS
        xlabel = r"r (cm)"
        ylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$"
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
		
        # show LEGEND
        ax1.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        # ax1.savefig(data_prefix+'mean_rhoX_'+xnucid+'.png')
		
    def plot_Xtransport_equation(self,xbl,xbr):
        pass
		
    def plot_Xflux(self,xbl,xbr):
        pass
		
    def plot_Xflux_equation(self,xbl,xbr):
        pass

    def plot_Xvariance(self,xbl,xbr):
        pass
		
    def plot_Xvariance_equation(self,xbl,xbr):
        pass		
		
    def plot_X_Ediffusivity(self,xbl,xbr):
    # Eulerian diffusivity
        pass
		
    def plot_X_Ldiffusivity(self,xbl,xbr):
    # Lagrangian diffusivity
        pass
				
    def gauss(x, *p): 
    # Define model function to be used to fit to the data above:
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))		
	
    def idx_bndry(self,xbl,xbr):
    # calculate indices of grid boundaries 
        rr = np.asarray(self.xzn0)
        xlm = np.abs(rr-xbl)
        xrm = np.abs(rr-xbr)
        idxl = int(np.where(xlm==xlm.min())[0])
        idxr = int(np.where(xrm==xrm.min())[0])	
        return idxl,idxr
	
	
