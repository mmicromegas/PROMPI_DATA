import PROMPI_data as prd 
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# class for plotting background stratification of PROMPI models from ransdat

class PROMPI_bgr(prd.PROMPI_ransdat,object):

    def __init__(self,filename):
        super(PROMPI_bgr,self).__init__(filename) 
        self.data = self.rans()

    def SetMatplotlibParams(self):
        """ This routine sets some standard values for matplotlib """ 
        """ to obtain publication-quality figures """

        # plt.rc('text',usetex=True)
        # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        plt.rc('font',**{'family':'serif','serif':['Times New Roman']})
        plt.rc('font',size=14.)
        plt.rc('lines',linewidth=2,markeredgewidth=2.,markersize=10)
        plt.rc('axes',linewidth=1.5)
        plt.rcParams['xtick.major.size']=8.
        plt.rcParams['xtick.minor.size']=4.
        plt.rcParams['figure.subplot.bottom']=0.15
        plt.rcParams['figure.subplot.left']=0.17		
        plt.rcParams['figure.subplot.right']=0.85		

    def idx_bndry(self,xbl,xbr):
        rr = np.asarray(self.data['xzn0'])
        xlm = np.abs(rr-xbl)
        xrm = np.abs(rr-xbr)
        idxl = int(np.where(xlm==xlm.min())[0])
        idxr = int(np.where(xrm==xrm.min())[0])	
        return idxl,idxr

    def plot_log_q1q2(self,xbl,xbr,f1,f2,xlabel_1,ylabel_1,ylabel_2,plabel_1,plabel_2):
        rr = np.asarray(self.data['xzn0'])		
        f_1 = self.data[f1]
        f_2 = self.data[f2]
		
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        to_plt1 = np.log10(f_1)
        to_plt2 = np.log10(f_2)
	
        fig, ax1 = plt.subplots(figsize=(7,6))
		
        ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.plot(rr,to_plt1,color='b',label = plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7,prop={'size':18})

        ax2 = ax1.twinx()
        ax2.axis([xbl,xbr,np.min(to_plt2[idxl:idxr]),np.max(to_plt2[idxl:idxr])])
        ax2.plot(rr, to_plt2,color='r',label = plabel_2)
        ax2.set_ylabel(ylabel_2)
        ax2.tick_params('y')
        ax2.legend(loc=1,prop={'size':18})
		
        plt.show(block=False)

    def plot_log_q1(self,xbl,xbr,f1,xlabel_1,ylabel_1,plabel_1):
        rr = np.asarray(self.data['xzn0'])		
        f_1 = self.data[f1]
		
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        to_plt1 = np.log10(f_1)
	
        fig, ax1 = plt.subplots(figsize=(7,6))
		
        ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.plot(rr,to_plt1,color='b',label = plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7,prop={'size':18})

        plt.show(block=False)

    def plot_lin_q1(self,xbl,xbr,f1,xlabel_1,ylabel_1,plabel_1):
        rr = np.asarray(self.data['xzn0'])		
        f_1 = self.data[f1]
		
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        to_plt1 = f_1
	
        fig, ax1 = plt.subplots(figsize=(7,6))
		
        ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.plot(rr,to_plt1,color='b',label = plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7,prop={'size':18})

        plt.show(block=False)
        
        
    def plot_lin_q1q2(self,xbl,xbr,f1,f2,xlabel_1,ylabel_1,ylabel_2,plabel_1,plabel_2):
        rr = np.asarray(self.data['xzn0'])		
        if (f1) == 'enuc':
            f_1 = self.data['enuc1']+self.data['enuc2']
            f_2 = self.data[f2]
        elif (f2) == 'enuc':
            f_1 = self.data[f1]
            f_2 = self.data['enuc1']+self.data['enuc2']
        else:
             self.data[f1]
             self.data[f2]
		
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        to_plt1 = f_1
        to_plt2 = f_2
	
        fig, ax1 = plt.subplots(figsize=(7,6))
		
        ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.plot(rr,to_plt1,color='b',label = plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7,prop={'size':18})

        ax2 = ax1.twinx()
        ax2.axis([xbl,xbr,np.min(to_plt2[idxl:idxr]),np.max(to_plt2[idxl:idxr])])
        ax2.plot(rr, to_plt2,color='r',label = plabel_2)
        ax2.set_ylabel(ylabel_2)
        ax2.tick_params('y')
        ax2.legend(loc=1,prop={'size':18})
		
        plt.show(block=False)
        

    def plot_x(self,xbl,xbr,rr,X):
        pass


    def GETRATEcoeff(self,reaction):

        cl = np.zeros(7)

        if (reaction == 'c12_plus_c12_to_p_na23_r'):

            cl[0] = +0.585029E+02 
            cl[1] = +0.295080E-01
            cl[2] = -0.867002E+02 
            cl[3] = +0.399457E+01
            cl[4] = -0.592835E+00
            cl[5] = -0.277242E-01
            cl[6] = -0.289561E+01

        if (reaction == 'c12_plus_c12_to_he4_ne20_r'):

            cl[0] = +0.804485E+02 
            cl[1] = -0.120189E+00
            cl[2] = -0.723312E+02 
            cl[3] = -0.352444E+02
            cl[4] = +0.298646E+01
            cl[5] = -0.309013E+00
            cl[6] = +0.115815E+02


        if (reaction == 'he4_plus_c12_to_o16_r'):

            cl[0] = +0.142191E+03 
            cl[1] = -0.891608E+02
            cl[2] = +0.220435E+04 
            cl[3] = -0.238031E+04
            cl[4] = +0.108931E+03
            cl[5] = -0.531472E+01
            cl[6] = +0.136118E+04

        if (reaction == 'he4_plus_c12_to_o16_nr'):

            cl[0] = +0.184977E+02 
            cl[1] = +0.482093E-02
            cl[2] = -0.332522E+02 
            cl[3] = +0.333517E+01
            cl[4] = -0.701714E+00
            cl[5] = +0.781972E-01
            cl[6] = -0.280751E+01

        if (reaction == 'o16_plus_o16_to_p_p31_r'):

            cl[0] = +0.852628E+02  
            cl[1] = +0.223453E+00
            cl[2] = -0.145844E+03 
            cl[3] = +0.872612E+01
            cl[4] = -0.554035E+00
            cl[5] = -0.137562E+00
            cl[6] = -0.688807E+01

        if (reaction == 'o16_plus_o16_to_he4_si28_r'):

            cl[0] = +0.972435E+02 
            cl[1] = -0.268514E+00
            cl[2] = -0.119324E+03 
            cl[3] = -0.322497E+02
            cl[4] = +0.146214E+01
            cl[5] = -0.200893E+00
            cl[6] = +0.132148E+02

        if (reaction == 'ne20_to_he4_o16_nv'):

            cl[0] =  +0.637915E+02
            cl[1] =  -0.549729E+02
            cl[2] =  -0.343457E+02
            cl[3] =  -0.251939E+02
            cl[4] =  +0.479855E+01
            cl[5] =  -0.146444E+01
            cl[6] =  +0.784333E+01

        if (reaction == 'ne20_to_he4_o16_rv'):

            cl[0] = +0.109310E+03 
            cl[1] = -0.727584E+02
            cl[2] = +0.293664E+03 
            cl[3] = -0.384974E+03
            cl[4] = +0.202380E+02
            cl[5] = -0.100379E+01
            cl[6] = +0.201193E+03

        if (reaction == 'si28_to_he4_mg24_nv1'):

            cl[0] = +0.522024E+03 
            cl[1] = -0.122258E+03
            cl[2] = +0.434667E+03 
            cl[3] = -0.994288E+03
            cl[4] = +0.656308E+02
            cl[5] = -0.412503E+01
            cl[6] = +0.426946E+03

        if (reaction == 'si28_to_he4_mg24_nv2'):

            cl[0] = +0.157580E+02 
            cl[1] = -0.129560E+03
            cl[2] = -0.516428E+02 
            cl[3] = +0.684625E+02
            cl[4] = -0.386512E+01
            cl[5] = +0.208028E+00
            cl[6] = -0.320727E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv1'):

            cl[0] = -0.906347E+01 
            cl[1] = -0.241182E+02
            cl[2] = +0.373526E+01 
            cl[3] = -0.664843E+01
            cl[4] = +0.254122E+00
            cl[5] = -0.588282E-02
            cl[6] = +0.191121E+01

        if (reaction == 'he4_plus_si28_to_p_p31_rv2'):

            cl[0] = +0.552169E+01  
            cl[1] = -0.265651E+02
            cl[2] = +0.456462E-08 
            cl[3] = -0.105997E-07
            cl[4] = +0.863175E-09
            cl[5] = -0.640626E-10
            cl[6] = -0.150000E+01

        if (reaction == 'he4_plus_si28_to_p_p31_rv3'):

            cl[0] = -0.126553E+01 
            cl[1] = -0.287435E+02
            cl[2] = -0.309775E+02 
            cl[3] = +0.458298E+02
            cl[4] = -0.272557E+01
            cl[5] = +0.163910E+00
            cl[6] = -0.239582E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv4'):

            cl[0] = +0.296908E+02 
            cl[1] = -0.330803E+02
            cl[2] = +0.553217E+02 
            cl[3] = -0.737793E+02
            cl[4] = +0.325554E+01
            cl[5] = -0.144379E+00
            cl[6] = +0.388817E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv5'):

            cl[0] = +0.128202E+02 
            cl[1] = -0.376275E+02
            cl[2] = -0.487688E+02 
            cl[3] = +0.549854E+02
            cl[4] = -0.270916E+01
            cl[5] = +0.142733E+00
            cl[6] = -0.319614E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv6'):

            cl[0] = +0.381739E+02  
            cl[1] = -0.406821E+02
            cl[2] = -0.546650E+02 
            cl[3] = +0.331135E+02
            cl[4] = -0.644696E+00
            cl[5] = -0.155955E-02
            cl[6] = -0.271330E+02

        if (reaction == 'he4_plus_o16_to_ne20_n'):

            cl[0] = +0.390340E+02  
            cl[1] = -0.358600E-01
            cl[2] = -0.343457E+02 
            cl[3] = -0.251939E+02
            cl[4] = +0.479855E+01
            cl[5] = -0.146444E+01
            cl[6] = +0.634333E+01

        if (reaction == 'he4_plus_o16_to_ne20_r'):

            cl[0] = +0.845522E+02  
            cl[1] = -0.178214E+02
            cl[2] = +0.293664E+03 
            cl[3] = -0.384974E+03
            cl[4] = +0.202380E+02
            cl[5] = -0.100379E+01
            cl[6] = +0.199693E+03

        if (reaction == 'he4_plus_ne20_to_mg24_n'):

            cl[0] = +0.321588E+02  
            cl[1] = -0.151494E-01
            cl[2] = -0.446410E+02 
            cl[3] = -0.833867E+01
            cl[4] = +0.241631E+01
            cl[5] = -0.778056E+00
            cl[6] = +0.193576E+01

        if (reaction == 'he4_plus_ne20_to_mg24_r'):

            cl[0] = -0.291641E+03  
            cl[1] = -0.120966E+02
            cl[2] = -0.633725E+02 
            cl[3] = +0.394643E+03
            cl[4] = -0.362432E+02
            cl[5] = +0.264060E+01
            cl[6] = -0.121219E+03

        if (reaction == 'mg24_to_he4_ne20_nv'):

            cl[0] = +0.569781E+02  
            cl[1] = -0.108074E+03
            cl[2] = -0.446410E+02 
            cl[3] = -0.833867E+01
            cl[4] = +0.241631E+01
            cl[5] = -0.778056E+00
            cl[6] = +0.343576E+01

        if (reaction == 'mg24_to_he4_ne20_rv'):

            cl[0] = -0.266822E+03  
            cl[1] = -0.120156E+03
            cl[2] = -0.633725E+02 
            cl[3] = +0.394643E+03
            cl[4] = -0.362432E+02
            cl[5] = +0.264060E+01
            cl[6] = -0.119719E+03


        if (reaction == 'p_plus_na23_to_he4_ne20_n'):

            cl[0] = +0.334868E+03  
            cl[1] = -0.247143E+00
            cl[2] = +0.371150E+02 
            cl[3] = -0.478518E+03
            cl[4] = +0.190867E+03
            cl[5] = -0.136026E+03
            cl[6] = +0.979858E+02

        if (reaction == 'p_plus_na23_to_he4_ne20_r1'):

            cl[0] = +0.942806E+02   
            cl[1] = -0.312034E+01
            cl[2] = +0.100052E+03 
            cl[3] = -0.193413E+03
            cl[4] = +0.123467E+02
            cl[5] = -0.781799E+00
            cl[6] = +0.890392E+02

        if (reaction == 'p_plus_na23_to_he4_ne20_r2'):

            cl[0] = -0.288152E+02  
            cl[1] = -0.447000E+00
            cl[2] = -0.184674E-09 
            cl[3] = +0.614357E-09
            cl[4] = -0.658195E-10
            cl[5] = +0.593159E-11
            cl[6] = -0.150000E+01


        if (reaction == 'he4_plus_si28_to_c12_ne20_r'):

            cl[0] = -0.307762E+03  
            cl[1] = -0.186722E+03
            cl[2] = +0.514197E+03 
            cl[3] = -0.200896E+03
            cl[4] = -0.642713E+01
            cl[5] = +0.758256E+00
            cl[6] = +0.236359E+03

        if (reaction == 'p_plus_p31_to_c12_ne20_r'):

            cl[0] = -0.266452E+03   
            cl[1] = -0.156019E+03
            cl[2] = +0.361154E+03 
            cl[3] = -0.926430E+02
            cl[4] = -0.998738E+01
            cl[5] = +0.892737E+00
            cl[6] = +0.161042E+03


        if (reaction == 'c12_plus_ne20_to_p_p31_r'):

            cl[0] = -0.268136E+03    
            cl[1] = -0.387624E+02
            cl[2] = +0.361154E+03 
            cl[3] = -0.926430E+02
            cl[4] = -0.998738E+01
            cl[5] = +0.892737E+00
            cl[6] = +0.161042E+03

        if (reaction == 'c12_plus_ne20_to_he4_si28_r'):

            cl[0] = -0.308905E+03    
            cl[1] = -0.472175E+02
            cl[2] = +0.514197E+03 
            cl[3] = -0.200896E+03
            cl[4] = -0.642713E+01
            cl[5] = +0.758256E+00
            cl[6] = +0.236359E+03

        if (reaction == 'he4_plus_ne20_to_p_na23_n'):

            cl[0] = +0.335091E+03     
            cl[1] = -0.278531E+02
            cl[2] = +0.371150E+02 
            cl[3] = -0.478518E+03
            cl[4] = +0.190867E+03
            cl[5] = -0.136026E+03
            cl[6] = +0.979858E+02

        if (reaction == 'he4_plus_ne20_to_p_na23_r1'):

            cl[0] = +0.945037E+02    
            cl[1] = -0.307263E+02
            cl[2] = +0.100052E+03 
            cl[3] = -0.193413E+03
            cl[4] = +0.123467E+02
            cl[5] = -0.781799E+00
            cl[6] = +0.890392E+02

        if (reaction == 'he4_plus_ne20_to_p_na23_r2'):

            cl[0] = -0.285920E+02    
            cl[1] = -0.280530E+02
            cl[2] = -0.184674E-09 
            cl[3] = +0.614357E-09
            cl[4] = -0.658195E-10
            cl[5] = +0.593159E-11
            cl[6] = -0.150000E+01

        return cl

    def GET1NUCtimescale(self,c1l,c2l,c3l,c4l,c5l,c6l,c7l,tt):

        temp09 = self.eht_tt[:,tt]*1.e-9
        rate = exp(c1l + c2l*(temp09**(-1.)) + c3l*(temp09**(-1./3.)) + c4l*(temp09**(1./3.)) + c5l*temp09 + c6l*(temp09**(5./3.)) + c7l*np.log(temp09))
        timescale = 1./rate

        return timescale

    def GET2NUCtimescale(self,c1l,c2l,c3l,c4l,c5l,c6l,c7l,yi,yj,yk,tt):

        temp09 = self.eht_tt[:,tt]*1.e-9
        rate = exp(c1l + c2l*(temp09**(-1.)) + c3l*(temp09**(-1./3.)) + c4l*(temp09**(1./3.)) + c5l*temp09 + c6l*(temp09**(5./3.)) + c7l*np.log(temp09))
        timescale = 1./(self.eht_dd[:,tt]*yj*yk*rate/yi)

        return timescale

    def GET3NUCtimescale(self,c1l,c2l,c3l,c4l,c5l,c6l,c7l,yi1,yi2,tt):

        temp09 = self.eht_tt[:,tt]*1.e-9
        rate = exp(c1l + c2l*(temp09**(-1.)) + c3l*(temp09**(-1./3.)) + c4l*(temp09**(1./3.)) + c5l*temp09 + c6l*(temp09**(5./3.)) + c7l*np.log(temp09))
        timescale = 1./(self.eht_dd[:,tt]*self.eht_dd[:,tt]*yi1*yi2*rate)

        return timescale


    def PlotNucEnergyGen(self,xbl,xbr):
        """Plot nuclear reaction timescales"""

        rc = np.asarray(self.data['xzn0'])/1.e8
        tt = self.data['tt']
        dd = self.data['dd']

        xc12 = self.data['x0004']
        xo16 = self.data['x0005']
        xne20 = self.data['x0006']
        xsi28 = self.data['x0009']
#        enuc = self.data['enuc1']+self.data['enuc2']
        enuc1 = self.data['enuc1']
        enuc2 = np.abs(self.data['enuc2']) 
 
        plt.figure(figsize=(7,6))

        lb = 1.e3
        ub = 1.e16

        plt.axis([xbl,xbr,lb,ub])

#       ne20 > he4 + o16 (photo-d: resonance)
        t9 = tt/1.e9 
#+ 4.e-2*self.eht_tt[:,tt]/1.e9


        cl = self.GETRATEcoeff(reaction='ne20_to_he4_o16_rv')
        rate_ne20_alpha_gamma = exp(cl[0] + cl[1]*(t9**(-1.)) + cl[2]*(t9**(-1./3.)) + cl[3]*(t9**(1./3.)) + cl[4]*t9 + cl[5]*(t9**(5./3.)) + cl[6]*np.log(t9))


#       he4 + ne20 > mg24 
        cl = self.GETRATEcoeff(reaction='he4_plus_ne20_to_mg24_r')
        rate_ne20_alpha_gamma_code = exp(cl[0] + cl[1]*(t9**(-1.)) + cl[2]*(t9**(-1./3.)) + cl[3]*(t9**(1./3.)) + cl[4]*t9 + cl[5]*(t9**(5./3.)) + cl[6]*np.log(t9))


#       o16 + o16 > p + p31 (resonance)        
#        xo16 = self.fht_xo16[:,tt]
        cl = self.GETRATEcoeff(reaction='o16_plus_o16_to_p_p31_r')
        rate_o16_o16_pchannel_r =  exp(cl[0] + cl[1]*(t9**(-1.)) + cl[2]*(t9**(-1./3.)) + cl[3]*(t9**(1./3.)) + cl[4]*t9 + cl[5]*(t9**(5./3.)) + cl[6]*np.log(t9))

#       o16 + o16 > he4 + si28 (resonance)        
#        xo16 = self.fht_xo16[:,tt]
        cl = self.GETRATEcoeff(reaction='o16_plus_o16_to_he4_si28_r')
        rate_o16_o16_achannel_r =  exp(cl[0] + cl[1]*(t9**(-1.)) + cl[2]*(t9**(-1./3.)) + cl[3]*(t9**(1./3.)) + cl[4]*t9 + cl[5]*(t9**(5./3.)) + cl[6]*np.log(t9))
        

#       c12 + c12 > p + na23 (resonance)        
        cl = self.GETRATEcoeff(reaction='c12_plus_c12_to_p_na23_r')
        rate_c12_c12_pchannel_r =  exp(cl[0] + cl[1]*(t9**(-1.)) + cl[2]*(t9**(-1./3.)) + cl[3]*(t9**(1./3.)) + cl[4]*t9 + cl[5]*(t9**(5./3.)) + cl[6]*np.log(t9))

#       c12 + c12 > he4 + ne20 (resonance)        
        cl = self.GETRATEcoeff(reaction='c12_plus_c12_to_he4_ne20_r')
        rate_c12_c12_achannel_r =  exp(cl[0] + cl[1]*(t9**(-1.)) + cl[2]*(t9**(-1./3.)) + cl[3]*(t9**(1./3.)) + cl[4]*t9 + cl[5]*(t9**(5./3.)) + cl[6]*np.log(t9))



        t9a = t9/(1.+0.0396*t9)
        c_tmp1 = (4.27e26)*(t9a**(5./6.))
        c_tmp2 = t9**(3./2.)
        c_e_tmp1 = -84.165/(t9a**(1./3.))
        c_e_tmp2 = -(2.12e-3)*(t9**3.)

        rate_c12_c12 = c_tmp1/c_tmp2*(np.exp(c_e_tmp1 + c_e_tmp2))

        o_tmp1 = 7.1e36/(t9**(2./3.))
        o_c_tmp1 = -135.93/(t9**(1./3.))
        o_c_tmp2 = -0.629*(t9**(2./3.))
        o_c_tmp3 = -0.445*(t9**(4./3.))
        o_c_tmp4 = +0.0103*(t9**2.)
        rate_o16_o16 = o_tmp1*np.exp(o_c_tmp1 + o_c_tmp2 + o_c_tmp3 + o_c_tmp4)
        
        n_tmp1 = 4.11e11/(t9**(2./3.))
        n_e_tmp1 = -46.766/(t9**(1./3.)) - (t9/2.219)**2.
        n_tmp2 = 1.+0.009*(t9**(1./3.))+0.882*(t9**(2./3.))+0.055*t9+0.749*(t9**(4./3.))+0.119*(t9**(5./3.))
        n_tmp3 = 5.27e3/(t9**(3./2.))
        n_e_tmp3 = -15.869/t9

        n_tmp4 = 6.51e3*(t9**(1./2.))
        n_e_tmp4 = -16.223/t9

        rate_alpha_gamma_cf88 = n_tmp1*np.exp(n_e_tmp1)*n_tmp2+n_tmp3*np.exp(n_e_tmp3)+n_tmp4*np.exp(n_e_tmp4)

        c1_c12  = 4.8e18
        c1_o16  = 8.e18
        c1_ne20 = 2.5e29
        c1_si28 = 1.8e28


        
        
        yc12sq  = (xc12/12.)**2. 
        yo16sq  = (xo16/16.)**2.
        yne20sq = (xne20/20.)**2.
    
        yo16  = xo16/16.


        lag = (3.e-3)*(t9**(10.5))
        lox = (2.8e-12)*(t9/2.)**33.
        lca = (4.e-11)*(t9**29.)
        lsi = 120.*(t9/3.5)**5.

        en_c12  = c1_c12*yc12sq*dd*(rate_c12_c12_achannel_r+rate_c12_c12_pchannel_r)
        en_c12_acf88  = c1_c12*yc12sq*dd*(rate_c12_c12)
        en_o16  = c1_o16*yo16sq*dd*(rate_o16_o16_achannel_r+rate_o16_o16_pchannel_r)
        en_o16_acf88 = c1_o16*yo16sq*dd*(rate_o16_o16)
        en_ne20 = c1_ne20*(t9**(3./2.))*(yne20sq/yo16)*rate_ne20_alpha_gamma_code*np.exp(-54.89/t9)
        en_ne20_acf88 = c1_ne20*(t9**(3./2.))*(yne20sq/yo16)*rate_ne20_alpha_gamma*np.exp(-54.89/t9)      
        en_ne20_hw = c1_ne20*(t9**(3./2.))*(yne20sq/yo16)*lag*np.exp(-54.89/t9)   
#        en_ne20_ini = c1_ne20*(t9**(3./2.))*(yne20sq_ini/yo16)*rate_ne20_alpha_gamma_code*np.exp(-54.89/t9)
        en_ne20_lag = c1_ne20*(t9**(3./2.))*(yne20sq/yo16)*lag*np.exp(-54.89/t9)
        en_si28 = c1_si28*(t9**3./2.)*xsi28*(np.exp(-142.07/t9))*rate_ne20_alpha_gamma_code
        en_si28_acf88 = c1_si28*(t9**3./2.)*xsi28*(np.exp(-142.07/t9))*rate_ne20_alpha_gamma

        plt.semilogy(rc,en_c12,label=r"$\dot{\epsilon}_{\rm nuc}$ (C$^{12}$)")
        plt.semilogy(rc,en_o16,label=r"$\dot{\epsilon}_{\rm nuc}$ (O$^{16}$)")
        plt.semilogy(rc,en_ne20,label=r"$\dot{\epsilon}_{\rm nuc}$ (Ne$^{20}$)")
        plt.semilogy(rc,en_si28,label=r"$\dot{\epsilon}_{\rm nuc}$ (Si$^{28}$)")
        plt.semilogy(rc,en_c12+en_o16+en_ne20+en_si28,label='total',color='k')
        plt.semilogy(rc,enuc1,color='m',linestyle='--',label='nuc d')
        plt.semilogy(rc,enuc2,color='r',linestyle='--',label='-neutrinos')		
#        plt.semilogy(rc,enuc1+enuc2,color='b',linestyle='--',label='nuc+neutrinos')	
		
        plt.legend(loc=1,prop={'size':13})

        plt.ylabel(r"$\dot{\epsilon}_{\rm nuc}$ \ (erg g$^{-1}$ s$^{-1}$)")
        plt.xlabel('r ($10^8$ cm)')

        axvline(x=5.65,color='k',linewidth=1)
        plt.show(block=False)
#        text(9.,1.e6,r"ob",fontsize=42,color='k')

#        savefig(data_prefix+'nuclear_energy_gen.eps')
    
