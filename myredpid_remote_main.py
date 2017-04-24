# -*- coding: utf-8 -*-
"""
MY RED PID REMOTE

This program acquire date from the red pitaya after an initial trigger pulse and 
saves the results to the hard disc

PID SETTINGS EXAMPLE:
* P and I are between 0 and 100
* error of pid is the error as number of points between set and actual index
* gain G simply multiplies the error, should be larger than 1 and considerably smaller than the buffer length (number of points)
	-> for G=1, P=100 and error = 1 we reach the maximal output of the PID
Created on Thu Apr 06 10:37:27 2017
@author: Gerhard Schunk
"""
import sys, os
#os._exit(00)
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import redpitaya_scpi as scpi
import time as tm
from matplotlib.pyplot import cm 
from itertools import cycle
#from redpid_lib import autocorr, crosscorr, smooth, tiltcorrect, read_user_input, get_error_max
import multiprocessing
import ctypes

def autocorr(x):
    "calculates the autocorrelation function for multiple purposes"
#    result = np.convolve(x, x, mode='same')
    result = np.convolve(x, np.flipud(x), mode='full')
    return result[result.size/2.:]

def crosscorr(x,y):
    "calculates the crosscorrelation function for multiple purposes"
#    result = np.convolve(x, x, mode='same')
    result = np.convolve(x, np.flipud(y), mode='full')
    return result

def read_user_input(newstdin,flag,pid_status,P_pid,I_pid,G_pid,O_pid):
    "waits for input from the console"
    newstdinfile = os.fdopen(os.dup(newstdin))

    while True:
        stind_checkout = newstdinfile.readline()
        stind_checkout = stind_checkout[:len(stind_checkout)-1]

        if (stind_checkout == "on"):
	        print('Start the lock')
	        pid_status.value = 1   

        if (stind_checkout == "off"):
	        print('Stop the lock')
	        pid_status.value = 0   

        if (stind_checkout[0] == "P"):
	        print('New P', stind_checkout[1:]) 
	        P_pid.value = int(stind_checkout[1:])    

        if (stind_checkout[0] == "I"):
	        print('New I', stind_checkout[1:]) 
	        I_pid.value = int(stind_checkout[1:])    

        if (stind_checkout[0] == "G"):
	        print('New gain', stind_checkout[1:]) 
	        G_pid.value = int(stind_checkout[1:])

        if (stind_checkout[0] == "O"):
	        print('New offset', stind_checkout[1:]) 
	        O_pid.value = int(stind_checkout[1:])

        if (stind_checkout == "exit"):
	        print('Stop the music')
	        flag.value = 1
	        newstdinfile.close()

	        break
        else:
	        continue

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    http://wiki.scipy.org/Cookbook/SignalSmooth
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)]


########################################################################
###MAIN Process
########################################################################
"includes plotting"
if __name__ == '__main__':
	print("\n\n--------------------------")
	print("WGMR PID LOCK\n")
	print("Start the lock with \"on\" and stop with \"off\"")
	print("Change P, I or G (e.g \"P10\") or end programm with \"exit\" ")
	print("--------------------------\n\n")
	
	tm.sleep(1.)	#to make the programm stable (on windows it...)
			
	###PROGRAMM SETTINGS, mainly for debugging
	do_interactive = 1 #only do interactive plotting for =1
	do_plot = 1 #only plots of =1
	do_loop_saving = 0 #saves every trace of the loop! Be careful
	do_save_plot = 1 # save plots of all traces
	do_pid = 1 #only calculates the pid for = 1, mainly for debugging
	do_print_output = 1 #show console output if 1, gets changed according to runtime
	update_period_s = 2. #uodate time for console output
	
	###THREADING SETTINGS
	datamanager = multiprocessing.Manager()
	datalist = datamanager.list()
	flag = multiprocessing.Value(ctypes.c_int, 0) #the flag is used to terminate the whole program, 0 is running
	pid_status = multiprocessing.Value(ctypes.c_int, 0) # PID is off for 0, on for 1

	#FREQUENCY SWEEP SETTINGS
	sweep_span_MHz = 1000.

	###SCPI DATA ACQUISION
	#best
	decimation = int(2**6)
	buff_len = 2**12 #2ms

#	decimation = int(2**10)
#	buff_len = 2**10 #2ms
	num_run = 10
	sample_rate_MHz = 125. / decimation
	delta_time_s_ms = 1. / sample_rate_MHz * 10**(-3) 

	#PID SETTINGS
	P_pid = multiprocessing.Value(ctypes.c_int, 10) #P part of pid
	I_pid = multiprocessing.Value(ctypes.c_int, 0) # i part of pid
	G_pid = multiprocessing.Value(ctypes.c_int, 1) #can also be negative for the lock to work, the pid_offset gets added afterwards
	O_pid = multiprocessing.Value(ctypes.c_int, 50) #offset for the pid output in per cent, usually 50,  start in the middle of the voltage setting

	I_pid_curr = 0 #Initial setting of I part of pid
	pid_error = 0 # that is the error signal for the pid between 0 and 1, comes from (index_setpoint - index_actual )
	pid_output = 0 #the pid is starting from 0, but there is an pid offset of 50 percent to start in the middle

	###FOLDER SETTINGS
	savename = 'test'
	pathname = os.getcwd()
	dir_list = [x[0] for x in os.walk(pathname)]
	for dir_name in dir_list:
		if not(dir_name.find("plots")) == -1:
			plots_path = dir_name + dir_name[-6]
	data_path = plots_path[:-7] + plots_path[-1] +  'data' + plots_path[-1] 
		
	###PLOT SETTINGES
	if do_plot == 1:
		title_font = {'fontname':'Arial', 'size':'12', 'color':'black', 'weight':'normal',
		'verticalalignment':'bottom'} # Bottom vertical alignment for more space
		#legend_font = {'family':'Times New Roman','size':'8'}
		title_font = {'size':'11'}
		axis_font = {'size':'11'}
		ticks_font = {'size':'11'}
		label_font = {'size':'11'}
		legend_font = {'size':'8'}

		if do_interactive == 1:
			plt.ion()

		fig1 = plt.figure(1,figsize=(22.5/2.53, 30./2.53))
		ax1 = fig1.add_subplot(311)
		ax2 = fig1.add_subplot(312)
		ax3 = fig1.add_subplot(313)
	
		fig1.subplots_adjust(left=0.25)
		fig1.subplots_adjust(bottom=0.25)
		fig1.subplots_adjust(top=0.90)
		fig1.subplots_adjust(right=0.95)
	
		for axis in ['top','bottom','left','right']:
			ax1.spines[axis].set_linewidth(1.)
			ax2.spines[axis].set_linewidth(1.)
			ax3.spines[axis].set_linewidth(1.)

		plt_title = ax1.set_title(' ')
		color=cycle(cm.jet(np.linspace(0,1,100)))

		ax1_hdl, = ax1.plot([],[])
		ax2_hdl, = ax2.plot([],[])
		ax3_hdl, = ax3.plot([],[])
		
		set_line_hdl = ax3.axvline(1,color='red')
		act_line_hdl = ax3.axvline(1,color='black')

		ax1.set_ylabel('Amplitude (V)',**axis_font)
		ax2.set_ylabel('Amplitude (V)',**axis_font)
		ax3.set_ylabel('Amplitude (norm.)',**axis_font)
		ax3.set_xlabel('Time (ms)',**axis_font)
	
	###REMOTE CONNECTION
	rp_s = scpi.scpi("10.64.11.12")
	
#	print '###START RECORDING###'
#	print 'decimation is', decimation, 
#	print 'Nr of runs', num_run
#	print 'Nr of samples', buff_len
#	print 'sample_rate_MHz', sample_rate_MHz
#	print 'sampling distance_ms', delta_time_s_ms

	###START PID
	###READ USER INPUT
	newstdin = sys.stdin.fileno()
	reading_process = multiprocessing.Process(target=read_user_input, args=(newstdin,flag,pid_status,P_pid,I_pid,G_pid,O_pid))
	reading_process.start()

	###START RECORDING
	rp_s.tx_txt('ACQ:RST')
	rp_s.tx_txt('ACQ:DEC '+str(decimation))
	rp_s.tx_txt('ACQ:TRIG:LEV 1')
	
	ini_time_s = tm.time()
	update_time_s = ini_time_s
	ite_meas = 0
#	for ite_meas in range(num_run):
	
	########################################################################
	###MAIN LOOP
	########################################################################
	while flag.value == 0:
		if flag == 1:
			break

		ite_meas = ite_meas + 1
		start_time_s = tm.time()
		
#		print(start_time_s - print_update_time_s)
		if start_time_s - update_time_s > update_period_s:	#check if console output
			do_print_output = 1
			update_time_s = tm.time()
		else:
			do_print_output = 0
			
		rp_s.tx_txt('ACQ:TRIG:DLY ' + str(int(buff_len-1000.)))
		rp_s.tx_txt('ACQ:START')
		tm.sleep(.05)	#pause to refresh buffer
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
		time_trace1_ms = np.asarray(time_trace1_ms) * delta_time_s_ms

	#	while 1:
	#	    rp_s.tx_txt('ACQ:TRIG:STAT?')
	#	    rcv = 	rp_s.rx_txt() 
	##	    print rcv
	#	    if rcv == 'TD':
	#	        break
	#	tm.sleep(.1)	#pause to refresh buffer

		###READ TACE 2
	#	rp_s.tx_txt('ACQ:SOUR2:DATA?')
		rp_s.tx_txt('ACQ:SOUR2:DATA:OLD:N? ' + str(int(buff_len)))
		buff_string = rp_s.rx_txt()
		buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
		buff = list(map(float, buff_string))
		y_trace2_V = np.asarray(buff)
		time_trace2_ms = range(len(y_trace2_V))
		time_trace2_ms = np.asarray(time_trace2_ms) * delta_time_s_ms
	
		###PROCESS TRACES
		smooth_pts = 100.
		time_trace1_ms = smooth(time_trace1_ms,smooth_pts)
		y_trace1_V = smooth(y_trace1_V,smooth_pts)
	#	y_trace1_V = tiltcorrect(y_trace1_V)
		y_trace1_V = y_trace1_V- np.min(y_trace1_V)
		y_trace1_V = np.abs(y_trace1_V)

		time_trace2_ms = smooth(time_trace2_ms,3)
		y_trace2_V = smooth(y_trace2_V,3)
		y_trace2_V = y_trace2_V- np.average(y_trace2_V)

		###GET TIMING
		stop_time_s = tm.time()
		rel_start_time = int(1000.*(start_time_s - ini_time_s))
		run_time_s = int(1000.*(stop_time_s - start_time_s))
		
		if do_print_output == 1:
			print("\n")
			print("start time")
			print(rel_start_time, " ms")
			print("run time")
			print(run_time_s, " ms")

		#SAVE DATA
		if do_loop_saving == 1:
			savename_loop = savename + '_trace1_nr_' +str(ite_meas)
		np.array(y_trace1_V).dump(open(data_path+'\\'+savename_loop+'.npy', 'wb'))
		#myArray = np.load(open('array.npy', 'rb'))
		
		
		if do_print_output == 1:
			print("flag, pid_status, P, I, G, O")
			print(flag.value, pid_status.value, P_pid.value, I_pid.value, G_pid.value, O_pid.value)
#		newstdin = sys.stdin.fileno()
#		read_user_input(newstdin)
	
		if pid_status.value == 0:
			y_trace_set_V = y_trace1_V # takes the setpoint 

		else: # starts locking, the PID values are calculated in per cent (between 0 and 100)
			crosscorr_V = crosscorr(y_trace1_V,y_trace_set_V)
			time_trace_cross = range(len(crosscorr_V))
			time_trace_cross_ms = np.asarray(time_trace_cross) * delta_time_s_ms
			ind_max_cross = np.argmax(crosscorr_V) # the maximum of the correlation between set_trace and actual trace gives the error
			pid_error_ind = ind_max_cross -buff_len
			pid_error = float(1.*pid_error_ind/buff_len) # error between 0 and 1

			###CALCULATE PID OUTPUT
			P_pid_curr = P_pid.value * pid_error # P value should between 0 and 100
			I_pid_curr = I_pid.value * pid_error + I_pid_curr # I value should between 0 and 100

			pid_output = P_pid_curr + I_pid_curr # pid_output should be within 0 and 100
			pid_output_with_gain = pid_output * G_pid.value

		###SET PID OUTPUT
		#rp_s.tx_txt('CR/LF'); #no idea what this one does
		if do_pid == 1:
			pid_output_percent = pid_output + O_pid.value # pid_offset let's the output start in the middle (50) to have the maximal range.
			if pid_output_percent < 0:
				pid_output_percent = 0
			elif pid_output_percent > 100.:
				pid_output_percent = 100.

			pid_output_V = str((1.8/100.)*pid_output_percent)     #from 0 - 1.8 volts
			pin_out_num = '2'  #Analog outputs 0,1,2,3
			scpi_command = 'ANALOG:PIN AOUT' + pin_out_num + ',' + pid_output_V
			rp_s.tx_txt(scpi_command)

			pid_error_MHz = pid_error * sweep_span_MHz

			if do_print_output == 1:
				av_time_ms = (1000.*(stop_time_s-ini_time_s ) / ite_meas)
				print("averaged run time of programm")
				print(av_time_ms, ' ms')

			if do_print_output == 1:
				print("pid_error (pcent), pid_error (MHz), pid_offset (pcent), pid_output (V)")
#				print(np.round(1000.*pid_error)/1000.,np.round(1000.*pid_error_MHz)/1000.,np.round(1000.*pid_offset)/1000.,np.round(1000.*pid_output_V)/1000.)
				print(pid_error,pid_error_MHz,O_pid.value,pid_output_V)

		###PLOT TRACES
		if do_plot == 1:
			plt_title = ax1.set_title('measured after ' + str(rel_start_time) + ' ms',**title_font)

			c=next(color)
#			ax1_hdl.set_xdata(time_trace1_ms)
#			ax1_hdl.set_ydata(y_trace1_V)

			ax1_hdl.remove()
			ax1_hdl, = ax1.plot(time_trace1_ms,y_trace1_V,markersize=5,linewidth = 1,color=c)
			ax1.set_xlim([0.,max(time_trace1_ms)])

			ax2_hdl.remove()
			ax2_hdl, = ax2.plot(time_trace2_ms,y_trace2_V,markersize=5,linewidth = 1,color=c)
			ax2.set_xlim([0.,max(time_trace2_ms)])

			if pid_status.value == 1:
				ax3_hdl.remove()
				set_line_hdl.remove()
				act_line_hdl.remove()
				ax3_hdl, = ax3.plot(time_trace_cross_ms,crosscorr_V,markersize=5,linewidth = 1,color=c)
				set_line_hdl = ax3.axvline(time_trace_cross_ms[ind_max_cross],color='red')
				act_line_hdl = ax3.axvline(time_trace_cross_ms[int(buff_len)],color='black')
				ax3.set_xlim([0.3*2.*max(time_trace2_ms),.7*2.*max(time_trace2_ms)])

			if do_interactive == 1:
				plt.draw()
#				tm.sleep(1.)

			###SAVE PLOTS
			if do_save_plot == 1 and do_print_output:
				fig1.canvas.draw()
				xlabels = [item.get_text() for item in ax1.get_xticklabels()]
				ax1.set_xticklabels(xlabels,**ticks_font)
				ylabels = [item.get_text() for item in ax1.get_yticklabels()]
				ax1.set_yticklabels(ylabels,**ticks_font)
#				if do_print_output == 1:
#					print("save plot in",plots_path + savename + ".png")

				#fig1.savefig(plots_path + '//' + savename + '.pdf', transparent=True)
				fig1.savefig(plots_path + savename_loop + ".png", dpi=300, transparent=True)

	if do_save_plot == 1:
		plt.close()

	print("Farewell, master!")
	sys.exit()
	