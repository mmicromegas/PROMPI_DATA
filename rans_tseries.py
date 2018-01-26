import PROMPI_data as pt
import numpy as np
import os
import matplotlib.pyplot as plt       
	   
datadir = './DATA/'
dataout = 'tseries_ransdat'
trange = [0.2,0.5]
tavg = 0.1

ransdat = [file for file in os.listdir(datadir) if "ransdat" in file]
ransdat = [file.replace(file,datadir+file) for file in ransdat]	

filename = ransdat[0]
ts = pt.PROMPI_ransdat(filename)
qqx = ts.rans_qqx()
ransl = ts.rans_list()
		
nstep = []
time  = []
dt  = []	
		
for filename in ransdat:
    print(filename)
    ts = pt.PROMPI_ransdat(filename)
    rans_tstart, rans_tend, rans_tavg = ts.rans_header()
    time.append(rans_tend)
    dt.append(rans_tavg)

# convert to array

time = np.asarray(time)
dt = np.asarray(dt)		
nt = len(ransdat)

print('Numer of snapshots: ', nt)
print('Available time range:',min(time),round(max(time),3))
print('Restrict data to time range:', trange[0],trange[1])

#   limit snapshot list to time range of interest

idx = np.where((time > trange[0]) & (time < trange[1]))
time = time[idx]
dt = dt[idx]		

#  time averaging window

timecmin = min(time)+tavg/2.0
timecmax = max(time)-tavg/2.0
itc      = np.where((time >= timecmin) & (time <= timecmax))
timec    = time[itc]
ntc      = len(timec)

print('Number of time averaged snapshots: ', ntc)
print('Averaged time range: ',round(timecmin,3), round(timecmax,3))
print('qqx',qqx)

# READ IN DATA

eh = []
for i in range(nt):
    filename = ransdat[i]
    ts = pt.PROMPI_ransdat(filename) 
    field = [[data for data in ts.rans()[s]] for s in ts.ransl]
    eh.append(field)

# eh = eh(r,time,quantity)	
# plt.plot(eh[:][nt-1][2])
# plt.show(block=False)

# TIME AVERAGING

eht = {}

for s in ts.ransl:
    idx = ts.ransl.index(s)
    tmp2 = []
    for i in range(ntc):
        itavg = np.where((time >= (timec[i]-tavg/2.)) & (time <= (timec[i]+tavg/2.)))
        sumdt = np.sum(dt[itavg])
        tmp1 = np.zeros(qqx)
        for j in itavg[0]:   
            tmp1 += np.asarray(eh[:][j][idx])*dt[j]
        tmp2.append(tmp1/sumdt)
    field = {str(s) : tmp2}  		
    eht.update(field)     

print(ts.ransl)

fld = 'tt'
a = eht[fld]
b = a[:][2]

#print(b)

fig, ax1 = plt.subplots(figsize=(7,6))

xbl = 3.72e8
xbr = 9.8e8

#fig, ax1 = plt.subplots(figsize=(7,6))

for i in range(nt):
    filename = ransdat[i]
    ts = pt.PROMPI_ransdat(filename)
    plt.plot(ts.rans()[fld])

ax1.plot(b,color='k')
	
plt.show(block=False)
	




 
