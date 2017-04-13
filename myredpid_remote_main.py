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

#############PRE SETTINGS
pathname = os.getcwd()
plots_path = pathname + '\\plots'
if not os.path.exists(plots_path):
	os.makedirs(plots_path)
	
title_font = {'fontname':'Arial', 'size':'12', 'color':'black', 'weight':'normal',
'verticalalignment':'bottom'} # Bottom vertical alignment for more space
#axis_font = {'fontname':'Times New Roman', 'size':'11'}
#ticks_font = {'fontname':'Times New Roman', 'size':'11'}
#label_font = {'family':'Times New Roman','size':'11'}
#legend_font = {'family':'Times New Roman','size':'8'}
axis_font = {'size':'11'}
ticks_font = {'size':'11'}
label_font = {'size':'11'}
legend_font = {'size':'8'}


fig1 = plt.figure(2,figsize=(8.5/2.53, 5./2.53))
ax1 = fig1.add_subplot(111)
fig1.subplots_adjust(left=0.20)
fig1.subplots_adjust(bottom=0.25)
fig1.subplots_adjust(top=0.95)
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


rp_s.tx_txt('ACQ:START')
rp_s.tx_txt('ACQ:TRIG NOW')

while 1:
    rp_s.tx_txt('ACQ:TRIG:STAT?')
    if rp_s.rx_txt() == 'TD':
        break

rp_s.tx_txt('ACQ:SOUR1:DATA?')
buff_string = rp_s.rx_txt()
buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
buff = list(map(float, buff_string))


##########PLOT RESULTS			
filename = 'oszi_trace'	
ax1.set_title(filename)
ax1.plot(buff,'k',markersize=5,linewidth = 1)
#ax1.set_xlim([0.,100.])
#ax1.set_ylim([0.7,1.05])

#############LABELS AND LEGENDS########################
#ax1.grid()

for axis in ['top','bottom','left','right']:
	ax1.spines[axis].set_linewidth(1.)

#ticks:
ax1.locator_params(axis = 'x', nbins = 6)
ax1.locator_params(axis = 'y', nbins = 4)

ax1.set_xlabel('Time (s)',**axis_font)
ax1.set_ylabel('Amplitude (V)',**axis_font)
ax1.grid()

locs,labels = plt.yticks()
#leg_handle = plt.legend(['spectrum'],loc='lower right',prop=legend_font)

fig1.canvas.draw()

xlabels = [item.get_text() for item in ax1.get_xticklabels()]
ax1.set_xticklabels(xlabels,**ticks_font)
ylabels = [item.get_text() for item in ax1.get_yticklabels()]
ax1.set_yticklabels(ylabels,**ticks_font)

fig1.savefig(plots_path + '//' + filename + '.pdf', transparent=True)
fig1.savefig(plots_path + '//' + filename + '.png', dpi=300, transparent=True)


print 'Farewell, master!'