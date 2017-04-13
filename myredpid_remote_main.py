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
import redpitaya_scpi as scpi
import time as tm


#############PRE SETTINGS
ini_time_ms = tm.time()
pathname = os.getcwd()
data_path = pathname + '\\data'
plots_path = pathname + '\\plots'
if not os.path.exists(plots_path):
	os.makedirs(plots_path)
	
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


fig1 = plt.figure(2,figsize=(8.5/2.53, 5./2.53))
ax1 = fig1.add_subplot(111)
fig1.subplots_adjust(left=0.25)
fig1.subplots_adjust(bottom=0.25)
fig1.subplots_adjust(top=0.90)
fig1.subplots_adjust(right=0.95)

###########REMOTE CONNECTION
rp_s = scpi.scpi("10.64.11.12")


###########DATA ACQUISION
#acquire voltage from different pins
#len_data = 4
#amp_V = np.zeros([10,1])
#for i in range(4):
#    rp_s.tx_txt('ANALOG:PIN? AIN' + str(i))
#    amp_V[i] = float(rp_s.rx_txt())
#    print ("Measured voltage on AI["+str(i)+"] = "+str(amp_V)+"V")

#Decimation,Sampling Rate,Time scale/length of a buffer,Trigger delay in samples,Trigger delay in seconds
#1,125 MS/s,131.072 us,from - 8192 to x,-6.554E-5 to x
#8,15.6 MS/s,1.049 ms,from - 8192 to x,-5.243E-4 to x
#64,1.9 MS/s,8.389 ms,from - 8192 to x,-4.194E-3 to x
#1024,122.0 MS/s,134.218 ms,from - 8192 to x,-6.711E-2 to x
#8192,15.2 kS/s,1.074 s,from - 8192 to x,-5.369E-1 to x
#65536,7.6 kS/s,8.590 s,from - 8192 to x,-4.295E+0 to x

decimation = int(2**10)

print 'start recording'
print 'decimation is', decimation

sample_rate_MHz = 125. / decimation
delta_time_ms_ms = 1. / sample_rate_MHz * 10**(-3) 
print 'sample_rate_MHz', sample_rate_MHz, ' sampling distance_ms', delta_time_ms_ms

#START RECORDING
rp_s.tx_txt('ACQ:RST')
rp_s.tx_txt('ACQ:DEC '+str(decimation))
rp_s.tx_txt('ACQ:TRIG:LEV 0.5')


for ite_meas in range(10):
	start_time_ms = tm.time()
	rp_s.tx_txt('ACQ:START')
#	rp_s.tx_txt('ACQ:TRIG NOW')
#	rp_s.tx_txt('ACQ:TRIG EXT_PE')
	rp_s.tx_txt('SOUR1:TRIG:SOUR EXT')	

#	while 1
#		trig_rsp=rp_s.('ACQ:TRIG:STAT?')
#   
#	if strcmp('TD',trig_rsp(1:2))  % Read only TD
#   
#		break
   

	while 1:
	    rp_s.tx_txt('ACQ:TRIG:STAT?')
	    rcv = 	rp_s.rx_txt() 
	    print rcv
	    if rcv == 'TD':
	        break
	
	rp_s.tx_txt('ACQ:SOUR1:DATA?')
	buff_string = rp_s.rx_txt()
	buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
	buff = list(map(float, buff_string))
	
	#evalutae data
	buff = np.asarray(buff)
	buff = buff- np.average(buff)
	time_ms = np.arange(0,len(buff)*delta_time_ms_ms,delta_time_ms_ms)
	
	#GET TIME
	stop_time_ms = tm.time()
	
	rel_start_time = int(1000.*(start_time_ms - ini_time_ms))
	run_time_ms = int(1000.*(stop_time_ms - start_time_ms))
	
	print 'relative start time', rel_start_time, ' ms'
	print 'run time', run_time_ms, ' ms'
	
	##########PLOT RESULTS	
	plt.cla()		
	filename = 'oszi_trace'	
	ax1.set_title('meas. at ' + str(rel_start_time) + ' ms',**title_font)
	ax1.plot(time_ms,buff,'k',markersize=5,linewidth = 1)
	#ax1.set_xlim([0.,100.])
	#ax1.set_ylim([0.7,1.05])
	
	#############LABELS AND LEGENDS########################
	#ax1.grid()
	
	for axis in ['top','bottom','left','right']:
		ax1.spines[axis].set_linewidth(1.)
	
	#ticks:
	ax1.locator_params(axis = 'x', nbins = 6)
	ax1.locator_params(axis = 'y', nbins = 4)
	
	ax1.set_xlabel('Time (ms)',**axis_font)
	ax1.set_ylabel('Amplitude (V)',**axis_font)
	ax1.grid()
	
	locs,labels = plt.yticks()
	#leg_handle = plt.legend(['spectrum'],loc='lower right',prop=legend_font)
	
	fig1.canvas.draw()
	
	xlabels = [item.get_text() for item in ax1.get_xticklabels()]
	ax1.set_xticklabels(xlabels,**ticks_font)
	ylabels = [item.get_text() for item in ax1.get_yticklabels()]
	ax1.set_yticklabels(ylabels,**ticks_font)
	
	#SAVE DATA
	#save data
#	savename_data = 'test_at_'+str(int((start_time_ms-ini_time_ms)*1000.)) + '_ms' ite_meas
	savename = str(ite_meas) + '_test'
	np.array(buff).dump(open(data_path+'\\'+savename+'.npy', 'wb'))
	#myArray = np.load(open('array.npy', 'rb'))
	
	#save plots
#	fig1.savefig(plots_path + '//' + savename + '.pdf', transparent=True)
	fig1.savefig(plots_path + '//' + savename + '.png', dpi=300, transparent=True)


print 'Farewell, master!'