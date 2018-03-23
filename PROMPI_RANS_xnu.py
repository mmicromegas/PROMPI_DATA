import numpy as np
import matplotlib.pyplot as plt
import CALCULUS as calc

class PROMPI_xnu(calc.CALCULUS,object):

    def __init__(self,filename,inuc,intc):
        super(PROMPI_xnu,self).__init__(filename) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.intc = intc
		
        # assign data and convert it to numpy array
        self.timec       = eht.item().get('timec')[intc] 
        self.xzn0        = np.asarray(eht.item().get('rr')) 
        self.eht_dd      = np.asarray(eht.item().get('dd')[intc])
        self.eht_ddxi    = np.asarray(eht.item().get('ddx'+inuc)[intc])
        self.eht_ddux    = np.asarray(eht.item().get('ddux')[intc])
        self.eht_ddxiux  = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        self.eht_ddxidot = np.asarray(eht.item().get('ddx'+inuc+'dot')[intc])	
        self.eht_ddxisq  = np.asarray(eht.item().get('ddx'+inuc+'sq')[intc])
		
        # store time series for time derivatives
        self.t_timec       = eht.item().get('timec') 
        self.t_eht_ddxi    = np.asarray(eht.item().get('ddx'+inuc))		
        self.t_eht_ddxisq  = np.asarray(eht.item().get('ddx'+inuc+'sq'))
		
        #######################
        # Xi TRANSPORT EQUATION 
		
        # LHS dq/dt 		
        self.dtddxi = self.dt(self.t_eht_ddxi,self.xzn0,self.t_timec,intc)
        # LHS div(ddXiux)
        self.divddxiux = self.Div(self.eht_ddxi*self.eht_ddux/self.eht_dd,self.xzn0)
		
        # RHS div(fxi) 
        self.divfxi = self.Div(self.eht_ddxiux - self.eht_ddxi*self.eht_ddux/self.eht_dd,self.xzn0) 
        # RHS ddXidot 
        self.ddXidot = self.eht_ddxidot 
        # res
        self.resXiTransport = - self.dtddxi - self.divddxiux - self.divfxi + self.ddXidot
		
        # END Xi TRANSPORT EQUATION
        #######################
		
    def plot_Xrho(self,xbl,xbr,inuc,data_prefix):
        """Plot Xrho stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(inuc)
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.eht_ddxi
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis
        plt.axis([xbl,xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
		
        # plot DATA 
        plt.title('xnucid '+str(xnucid))
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho} \widetilde{X}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_rhoX_'+xnucid+'.png')
	
    def plot_Xtransport_equation(self,xbl,xbr,inuc,data_prefix):
        """Plot Xrho stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(inuc)
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtddxi
        lhs1 = - self.divddxiux 
		
        rhs0 = - self.divfxi
        rhs1 = + self.ddXidot
		
        res = - self.resXiTransport
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),np.min(rhs0[idxl:idxr]),np.min(rhs1[idxl:idxr]),np.min(res[idxl:idxr])])
        maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),np.max(rhs0[idxl:idxr]),np.max(rhs1[idxl:idxr]),np.max(res[idxl:idxr])])
        plt.axis([xbl,xbr,minx,maxx])
		
        # plot DATA 
        plt.title('xnucid '+str(xnucid))
        plt.plot(grd1,lhs0,color='r',label = r'$-\partial_t (\overline{\rho} \widetilde{X})$')
        plt.plot(grd1,lhs1,color='cyan',label = r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$')		
        plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f$')
        plt.plot(grd1,rhs1,color='g',label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_rhoX_'+xnucid+'.png')

		
    def plot_Xflux(self,xbl,xbr,inuc,data_prefix):
        """Plot Xflux stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(inuc)
		
        # load x GRID
        grd1 = self.xzn0		
		
        # load and calculate DATA to plot
        plt1 = self.eht_ddxiux - self.eht_ddxi*self.eht_ddux/self.eht_dd
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format Y AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))
		
        # limit x/y axis
        plt.axis([xbl,xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
		
        # plot DATA 
        plt.title('xnucid '+str(xnucid))
        plt.plot(grd1,plt1,color='k',label = r'f'+str(inuc))

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_Xflux_'+xnucid+'.png')
		
    def plot_Xflux_equation(self,xbl,xbr):
        pass

    def plot_Xvariance(self,xbl,xbr,inuc,data_prefix):
        """Plot Xvariance stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(inuc)
		
        # load x GRID
        grd1 = self.xzn0
		
        # load and calculate DATA to plot
        plt1 = (self.eht_ddxisq - self.eht_ddxi*self.eht_ddxi/self.eht_dd)/self.eht_dd
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format Y AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))
		
        # limit x/y axis
        plt.axis([xbl,xbr,1.e-20,np.max(plt1[idxl:idxr])])
		
        # plot DATA 
        plt.title('xnucid '+str(xnucid))
        plt.semilogy(grd1,plt1,color='b',label = r'$\sigma$'+str(inuc))

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{X''_i X''_i}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_Xvariance_'+xnucid+'.png')
		
	
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
	
	
