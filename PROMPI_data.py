import numpy as np

class PROMPI_ransdat:

    def __init__(self,filename):

#        tdata = open(ftycho,'r')

#        t_line1 = tdata.readline().split()
#        nspec = int(t_line1[1])

#        xnuc = []
#        for i in range(nspec):
#            xnuc.append(tdata.readline().split()[0])    
        
#        tdata.close()
        
        fhead = open(filename.replace("ransdat","ranshead"),'r') 

        header_line1 = fhead.readline().split()
        header_line2 = fhead.readline().split()
        header_line3 = fhead.readline().split()
        header_line4 = fhead.readline().split()

        self.nstep       = int(header_line1[0])
        self.rans_tstart = float(header_line1[1])
        self.rans_tend   = float(header_line1[2])
        self.rans_tavg   = float(header_line1[3])
		
        self.qqx    = int(header_line2[0])
        self.qqy    = int(header_line2[1])
        self.qqz    = int(header_line2[2])
        self.nnuc   = int(header_line2[3])
        self.nrans  = int(header_line2[4])
        ndims = [4,self.nrans,self.qqx]

        self.ransl = []		
        for line in range(self.nrans):
            line = fhead.readline().strip()
            self.ransl.append(line)

#        for inuc in range(nspec):
#            self.ransl = [field.replace(str(inuc+1),str(xnuc[inuc])) for field in self.ransl]
#            self.ransl = [field.replace("0","") for field in self.ransl]    

            
#        print(self.ransl)
            
        xznl = [] 
        xzn0 = []
        xznr = []
        yznl = []
        yzn0 = []
        yznr = [] 
        zznl = []
        zzn0 = []
        zznr = []
			
        for line in range(self.qqx):
            line = fhead.readline().strip()        
            xznl.append(float(line[8:22].strip()))
            xzn0.append(float(line[23:38].strip()))
            xznr.append(float(line[39:54].strip()))
			
        for line in range(self.qqy):
            line = fhead.readline().strip()        
            yznl.append(float(line[8:22].strip()))
            yzn0.append(float(line[23:38].strip()))
            yznr.append(float(line[39:54].strip()))	

        for line in range(self.qqz):
            line = fhead.readline().strip()
            zznl.append(float(line[8:22].strip()))
            zzn0.append(float(line[23:38].strip()))
            zznr.append(float(line[39:54].strip()))	

        frans = open(filename,'rb')
        self.data = np.fromfile(frans)		
#        self.data = np.fromfile(frans,dtype='>f',count=ndims[0]*ndims[1]*ndims[2])
        self.data = np.reshape(self.data,(ndims[0],ndims[1],ndims[2]),order='F')	

        self.ransd = {}
		
        self.ransd = {"xzn0" : xzn0}
		
        i = 0
#        print(self.ransl)
		
        for s in self.ransl:
            field = {str(s) : self.data[2,i,:]}
            self.ransd.update(field)
            i += 1
		
        frans.close()
        fhead.close()		
 
    def rans_header(self):
        return self.rans_tstart,self.rans_tend,self.rans_tavg
 
    def rans(self):	
        return self.ransd
		
    def rans_list(self):
        return self.ransl
		
    def rans_qqx(self):
        return self.qqx

    def ranslist(self):
        return np.asarray(self.xzn0)

    def ransdict(self):
        print self.eh.keys()
		
    def sterad(self):
        pass
    
class PROMPI_blockdat:

    def __init__(self,filename):
        pass
