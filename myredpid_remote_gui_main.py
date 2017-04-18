from PyQt4 import QtGui, QtCore
from Vitaminator_ui_V5 import MainWindow
import Tkinter
import tkFileDialog
import thread
import threading

def printyes():
    print 'yes man'
#        self.mplwidget.

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

    
class Window(MainWindow):
    def __init__(self):
        MainWindow.__init__(self)

#       Define constants
        self.A0 = 100
        self.threshold = 80
        self.numoftimezones = 4
        fig = self.mplwidget.figure
        self.ax1 = fig.add_subplot(111)
        fig.set_facecolor("white")
        fig.subplots_adjust(top=0.91)
        self.plot_diagramm()
        
#       shape gui
        self.pushButton_premium.close();
#       add own connections

        QtGui.QApplication.setStyle("cleanlooks")
        QtCore.QObject.connect(self.pushButton_plot__savedia, QtCore.SIGNAL(_fromUtf8("clicked()")), self.save_diagramm)
        QtCore.QObject.connect(self.horizontalSlider_temp_1, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.plot_diagramm)
        QtCore.QObject.connect(self.horizontalSlider_time_1, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.plot_diagramm)
        QtCore.QObject.connect(self.horizontalSlider_time_2, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.plot_diagramm)
        QtCore.QObject.connect(self.horizontalSlider_temp_2, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.plot_diagramm)
        QtCore.QObject.connect(self.horizontalSlider_temp_3, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.plot_diagramm)
        QtCore.QObject.connect(self.horizontalSlider_time_3, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.plot_diagramm) 
        QtCore.QObject.connect(self.horizontalSlider_temp_4, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.plot_diagramm)
        QtCore.QObject.connect(self.horizontalSlider_time_4, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.plot_diagramm) 
    
    def getfoldername(self,root):
        root = Tkinter.Tk()
        root.withdraw()
        savefolder = tkFileDialog.askdirectory(title='Dear employee of Nutricia - please choose a directoy for the save files.')
        print 'tkinter savename', savefolder
        root.mainloop()
        root.destroy()
        return savefolder

    
    def save_diagramm(self):
#        thread.start_new_thread(threadmain())

#        root = Tkinter.Tk()
#        root.withdraw()    
#        savefolder = tkFileDialog.askdirectory(title='Dear employee of Nutricia - please choose a directoy for the save files.')

#        root.mainloop()

        savefolder = 'C:\Users\gschunk.MPL\Desktop'
        savefilename = savefolder + '\\Vitamin_decay'
    
        T_C, t_days = [], []
        for k in range(self.numoftimezones):
            Tvalue = eval("self.horizontalSlider_temp_"+str(k+1)+".sliderPosition()")
            T_C.append(Tvalue)
            tvalue = eval("self.horizontalSlider_time_"+str(k+1)+".sliderPosition()")
            t_days.append(tvalue)
            savefilename = savefilename +'__T'+str(k)+'_'+str(Tvalue)+'_t'+str(k)+'_'+str(tvalue)
            
        thread.start_new_thread(self.mplwidget.print_pdf(savefilename+'.pdf'))
#        self.mplwidget.print_pdf(savefilename+'.pdf')
        
        delta_t_days,At = self.calc_decay()

#        save plot data
        f1 = open(savefilename + ".dat","a")
        f1.truncate()
        f1.write('Time_(days)'+'\t'+'Vit_conc_(percent)'+'\n')
        for int in range(len(delta_t_days)):
            entry = str(delta_t_days[int]) + '\t'+str(At[int])+'\n'
            f1.write(entry)  
        self.pushButton_plot__savedia.setText(QtGui.QApplication.translate("MainWindow", "DONE! now restart program", None, QtGui.QApplication.UnicodeUTF8))
        
    def calc_decay(self):
        T_C, t_days = [], []
        for k in range(self.numoftimezones):
            Tvalue = eval("self.horizontalSlider_temp_"+str(k+1)+".sliderPosition()")
            T_C.append(Tvalue)
            tvalue = eval("self.horizontalSlider_time_"+str(k+1)+".sliderPosition()")
            t_days.append(tvalue)            
        
        tmax_days = 0
        for t in t_days:
            tmax_days = t + tmax_days
            
        At = []
        delta_t_days = range(0,tmax_days*10)
        delta_t_days = tuple(float(x)/10 for x in delta_t_days)

        
        At_start = self.A0
        delta_t_days_start = 0
        
        timezone = 0
        for ite in range(len(delta_t_days)):
            if  delta_t_days[ite] <= (t_days[timezone] + delta_t_days_start):
                At_run = self.vitamin_decay(At_start,T_C[timezone],delta_t_days[ite]-delta_t_days_start)
                At.append(At_run)
            else:
                At_start = At_run
                delta_t_days_start = delta_t_days[ite-1]
                timezone = timezone + 1
                At_run = self.vitamin_decay(At_start,T_C[timezone],delta_t_days[ite]-delta_t_days_start)
                At.append(At_run)       
        return delta_t_days, At
    
        
    def plot_diagramm(self):
        label_font = {'family':'Times New Roman','size':'18'}
        ticks_font = {'family':'Times New Roman','size':'15'}
        legend_font = {'family':'Times New Roman','size':'10'}
#        self.pushButton_plot__savedia.setText(QtGui.QApplication.translate("MainWindow", "Save diagramm and data", None, QtGui.QApplication.UnicodeUTF8))

        delta_t_days, At = self.calc_decay()
        
        t_treshold_days = 0
        for ite in range(len(At)):
            if self.threshold > At[ite] and t_treshold_days == 0:
                t_treshold_days = delta_t_days[ite]
        
        t1_days = self.horizontalSlider_time_1.sliderPosition()
        t2_days = self.horizontalSlider_time_2.sliderPosition()
        t3_days = self.horizontalSlider_time_3.sliderPosition()   
        
        self.ax1.plot()
        self.ax1.hold(True)
        
        self.ax1.plot(delta_t_days,At,'k',linewidth=2,label = 'exponential decay')
        self.ax1.axvline(t1_days,linewidth=1, color="b")
        self.ax1.axvline(t2_days+t1_days,linewidth=1, color="g")
        self.ax1.axvline(t3_days+t2_days+t1_days,linewidth=1, color="m")
        
        if t_treshold_days == 0:
            label_theshold = str(self.threshold) + '% threshold'
        else:
            label_theshold = str(self.threshold) + '% threshold after ' + str(t_treshold_days) + ' days'
        self.ax1.axhline(self.threshold,linewidth=3, color="r",label = label_theshold)
        self.ax1.plot(delta_t_days,At,'k',linewidth=2,)
        
        self.ax1.legend()
        self.ax1.grid()
        ylimit = round(At[-1] * 0.8 / 10) * 10
        if ylimit <= 0:
            ylimit = 0
        self.ax1.set_ylim([ylimit,self.A0]) 
        
        self.ax1.set_xlabel('Time (days)',**label_font) 
        self.ax1.set_ylabel('Vitamin concentration (%)',**label_font) 
        self.ax1.set_title('Vitaminator 2.0 - calculate the shelf life',**label_font)
        self.ax1.figure.canvas.draw()
        xlabels = [item.get_text() for item in self.mplwidget.axes.get_xticklabels()]
        self.mplwidget.axes.set_xticklabels(xlabels,**ticks_font)
        ylabels = [item.get_text() for item in self.mplwidget.axes.get_yticklabels()]
        self.mplwidget.axes.set_yticklabels(ylabels,**ticks_font)
        self.ax1.figure.canvas.draw()
        self.ax1.hold(False)
    
    def vitamin_decay(self,A0,T_C,delta_t):
        import math
        delta_t = delta_t/30.
        gamma = (2.64*10**14)*math.exp(-11144./(T_C + 273.15))
        At = A0*math.exp(-gamma*delta_t)
        return At
        
    def vitamin_decay_inv(self,A0,At,T_C):
        import math
        gamma = (2.64*10**14)*math.exp(-11144./(T_C + 273.15))
        delta_t = -math.log(At/A0) / gamma * 30.
        return delta_t
        
if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    
######################GUI#################
#from matplotlibwidget import MatplotlibWidget
#import nutricialogo_rc
#
#class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
#    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
#        QtGui.QMainWindow.__init__(self, parent, f)
#        self.setupUi(self)
