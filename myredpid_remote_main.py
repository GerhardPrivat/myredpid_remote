# -*- coding: utf-8 -*-
"""
MY RED PID REMOTE

This program acquire date from the red pitaya after an initial trigger pulse and 
saves the results to the hard disc

Created on Thu Apr 06 10:37:27 2017
@author: Gerhard Schunk
"""
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import pdb
import redpitaya_scpi as scpi
import time as tm
from matplotlib.pyplot import cm 
from itertools import cycle
from redpid_lib import 

#############PRE SETTINGS
pathname = os.getcwd()
data_path = pathname + '\\data'
plots_path = pathname + '\\plots'
if not os.path.exists(plots_path):
	os.makedirs(plots_path)
	
#INITIALIZE PLOT
title_font = {'fontname':'Arial', 'size':'12', 'color':'black', 'weight':'normal',
'verticalalignment':'bottom'} # Bottom vertical alignment for more space
#axis_font = {'fontname':'Times New Roman', 'size':'11'}
#ticks_font = {'fontname':'Times New Roman', 'size':'11'}
#label_font = {'family':'Times New Roman','size':'11'}
#legend_font = {'family':'Times New Roman','size':'8'}
title_font = {'size':'11'}
axis_font = {'size':'11'}
ticks_font = {'size':'11'}
label_font = {'size':'11'}
legend_font = {'size':'8'}

plt.clf()
fig1 = plt.figure(1,tight_layout=True,figsize=(22.5/2.53, 10./2.53))
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)
fig1.subplots_adjust(left=0.25)
fig1.subplots_adjust(bottom=0.25)
fig1.subplots_adjust(top=0.90)
fig1.subplots_adjust(right=0.95)

for axis in ['top','bottom','left','right']:
	ax1.spines[axis].set_linewidth(1.)
	ax2.spines[axis].set_linewidth(1.)

plt_title = ax1.set_title(' ')

color=cycle(cm.rainbow(np.linspace(0,1,20)))

ax1_tace1, = ax1.plot([],[])
ax2_tace2, = ax2.plot([],[])

ax1.set_ylabel('Amplitude (V)',**axis_font)
ax2.set_ylabel('Amplitude (V)',**axis_font)
ax2.set_xlabel('Time (ms)',**axis_font)

ax1.grid()
ax2.grid()

###########REMOTE CONNECTION
rp_s = scpi.scpi("10.64.11.12")


###DATA ACQUISION
#acquire voltage from different pins
#len_data = 4
#amp_V = np.zeros([10,1])
#for i in range(4):
#    rp_s.tx_txt('ANALOG:PIN? AIN' + str(i))
#    amp_V[i] = float(rp_s.rx_txt())
#    print ("Measured voltage on AI["+str(i)+"] = "+str(amp_V)+"V")

#Decimation,Sampling Rate,Time scale/length of a buffer,Trigger delay in samples,Trigger delay in seconds
#1,125 MS/s                  ,131.072 us,from - 8192 to x,-6.554E-5 to x
#8,15.6 MS/s                ,1.049 ms,from - 8192 to x,-5.243E-4 to x
#64 = 2** 6 ,1.9 MS/s                 ,8.389 ms,from - 8192 to x,-4.194E-3 to x
#1024=2**10,122.0 MS/s 	,134.218 ms,from - 8192 to x,-6.711E-2 to x
#8192=2**13,15.2 kS/s 	      ,1.074 s,from - 8192 to x,-5.369E-1 to x
#65536,7.6 kS/s              ,8.590 s,from - 8192 to x,-4.295E+0 to x

#decimation = int(2**10) # max around 134 ms
#decimation = int(2**7) # max around 16 ms -> strange modes
decimation = int(2**6) # max around 8.389 ms for 2**14 = 16384 pts
buff_len = 2**13
#buff_len = 2**4
num_run = 10
sample_rate_MHz = 125. / decimation
delta_time_ms_ms = 1. / sample_rate_MHz * 10**(-3) 

print '###START RECORDING###'
print 'decimation is', decimation, 
print 'Nr of runs', num_run
print 'Nr of samples', buff_len
print 'sample_rate_MHz', sample_rate_MHz
print 'sampling distance_ms', delta_time_ms_ms

###START RECORDING
rp_s.tx_txt('ACQ:RST')
rp_s.tx_txt('ACQ:DEC '+str(decimation))
rp_s.tx_txt('ACQ:TRIG:LEV 1')

ini_time_ms = tm.time()
#for ite_meas in range(num_run):
while True:
	ite_meas = 1
	start_time_ms = tm.time()
	rp_s.tx_txt('ACQ:TRIG:DLY ' + str(int(buff_len-1000.)))
	rp_s.tx_txt('ACQ:START')
#	tm.sleep(1.)	#pause to refresh buffer
	rp_s.tx_txt('ACQ:TRIG EXT_PE')

	
	while 1:
	    rp_s.tx_txt('ACQ:TRIG:STAT?')
	    rcv = 	rp_s.rx_txt() 
#	    print rcv
	    if rcv == 'TD':
	        break
								
	###READ TACE 1
#	rp_s.tx_txt('ACQ:SOUR1:DATA?')
	rp_s.tx_txt('ACQ:SOUR1:DATA:OLD:N? ' + str(int(buff_len)))
	buff_string = rp_s.rx_txt()
	buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
	buff = list(map(float, buff_string))
	y_trace1_V = np.asarray(buff)
	time_trace1_ms = range(len(y_trace1_V))
	time_trace1_ms = np.asarray(time_trace1_ms) * delta_time_ms_ms

	###READ TACE 2
#	rp_s.tx_txt('ACQ:SOUR2:DATA?')
	rp_s.tx_txt('ACQ:SOUR2:DATA:OLD:N? ' + str(int(buff_len)))
	buff_string = rp_s.rx_txt()
	buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
	buff = list(map(float, buff_string))
	y_trace2_V = np.asarray(buff)
	time_trace2_ms = range(len(y_trace2_V))
	time_trace2_ms = np.asarray(time_trace2_ms) * delta_time_ms_ms

	###PROCESS DATA
	y_trace1_V = y_trace1_V- np.average(y_trace1_V)
	y_trace2_V = y_trace2_V- np.average(y_trace2_V)
	

	stop_time_ms = tm.time()
	
	rel_start_time = int(1000.*(start_time_ms - ini_time_ms))
	run_time_ms = int(1000.*(stop_time_ms - start_time_ms))
	
	print '\nNew trace:'
	print 'Relative start time', rel_start_time, ' ms'
	print 'run time', run_time_ms, ' ms'
			
	if 1==1:
		###PLOT TRACES	
		plt.ion()		
		c=next(color)
		
		plt_title = ax1.set_title('measured after ' + str(rel_start_time) + ' ms',**title_font)
		
		ax1_tace1.remove()
		ax1_tace1, = ax1.plot(time_trace1_ms,y_trace1_V,markersize=5,linewidth = 1,color=c)
		ax1.set_xlim([0.,max(time_trace1_ms)])
	
		ax2_tace2.remove()
		ax2_tace2, = ax2.plot(time_trace2_ms,y_trace2_V,markersize=5,linewidth = 1,color=c)
		ax2.set_xlim([0.,max(time_trace2_ms)])
		
		plt.draw()
		
	#SAVE DATA
	savename = str(ite_meas) + '_test'
#	np.array(y_trace1_V).dump(open(data_path+'\\'+savename+'.npy', 'wb'))
	#myArray = np.load(open('array.npy', 'rb'))

av_time = int(1000.*(stop_time_ms-ini_time_ms ) / num_run)
print 'averaged run time is ', av_time, ' ms'
###LABELS AND LEGENDS

#TICKS
ax1.locator_params(axis = 'x', nbins = 6)
ax1.locator_params(axis = 'y', nbins = 4)
	
###SAVE PLOTS
filename = 'oszi_trace'
fig1.canvas.draw()
xlabels = [item.get_text() for item in ax1.get_xticklabels()]
ax1.set_xticklabels(xlabels,**ticks_font)
ylabels = [item.get_text() for item in ax1.get_yticklabels()]
ax1.set_yticklabels(ylabels,**ticks_font)

#fig1.savefig(plots_path + '//' + savename + '.pdf', transparent=True)
fig1.savefig(plots_path + '//' + savename + '.png', dpi=300, transparent=True)

print 'Farewell, master!'