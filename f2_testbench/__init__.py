#from joblib import Parallel, delayed
import numpy as np
import matplotlib as mpl 
mpl.use('Agg') #To enble plotting without X
import matplotlib.pyplot as plt
from thesdk import *
import multiprocessing
from joblib import Parallel, delayed
from signal_generator_802_11n import *
from f2_util_classes import *

from f2_chip import *
from f2_channel import *
from f2_testbench.analyzers_mixin import *

class f2_testbench(thesdk,analyzers_mixin):
    Rs=160e6      #Baseband sampling frequency
    Rs_dsp=20e6   #Baseband sampling frequency
    frequency=1e9 #Rf center frequency 

    #Baseband signal dictionaries are used to control the signal generator
    #-----Sinusoidal test signals
    # Check if working. Receiver is now fiexed to OFDM
    bbsigdict_sinusoid={ 'mode':'sinusoid', 'freqs':[1.0e6 ], 'length':2**14, 'BBRs':40e6 };
    bbsigdict_sinusoid3= { 'mode':'sinusoid', 'freqs':[1.0e6 , 3e6, 7e6 ], 'length':2**14, 'BBRs':20e6 };
    bbsigdict_ofdm_sinusoid3={ 'mode':'ofdm_sinusoid', 'freqs':[-7.0e6 , 2e6, 5e6 ], 'length':2**14, 'BBRs':20e6 };
    
    #-----Data signals
    bbsigdict_randombitstream_QAM4_OFDM={ 'mode':'ofdm_random_qam', 'QAM':4, 'length':2**14, 'BBRs': 20e6 };

    bbsigdict_802_11n_random_QAM16_OFDM={ 'mode':'ofdm_random_802_11n', 'QAM':16, 'length':2**14, 'BBRs': 20e6 };

    #Channel dictionaries are used to16control the channel model
    channeldict_buffer= { 'model': 'lossless' , 'frequency': 1e9 }
    channeldict_buffer_1km= { 'model': 'lossless', 'frequency': 1e9, 'distance': 1000}
    #channeldict_lossless_awgn= { 'model': 'awgn', 'frequency': 1e9, 'bandwidth': 100e6, 'distance': 0 }

    channeldict_802_11n_A= { 'model': 'A',  'distance': 1000}
    channeldict_802_11n_B= { 'model': 'B',  'distance': 1000}
    channeldict_802_11n_C= { 'model': 'C',  'distance': 1000}
    channeldict_802_11n_D= { 'model': 'D',  'distance': 1000}
    channeldict_802_11n_E= { 'model': 'E',  'distance': 1000}
    channeldict_802_11n_F= { 'model': 'F',  'distance': 1000}

    def __init__(self,*arg):
        self.picpath=[]
        #Signals should be in form s(user,time,Txantenna)
        self.Txantennas=1                       #All the antennas process the same data
        self.Txpower=30                         #Output power per antenna in dBm
        self.Rxantennas=4
        self.Users=2
        self.Nbits=10  #ADC bits
        self.Txbits=9  #DAC bits
        self.Channeldir='Uplink'
        self.CPUBFmode = 'ZF';                  #['BF' | 'ZF'] Default to beam forming 
        self.DSPmode = 'cpu';                   #['cpu' | 'local'] bamforming can be local 
        self.dsp_decimator_model='py'
        self.dsp_decimator_scales=[1,1,1,1]
        self.dsp_decimator_cic3shift=12
        self.noisetemp=290
        self.nserdes=2
        self.Rs=160e6
        self.Rs_dsp=20e6
        self.Hstf=1                             #Synchronization filter
        self.bbsigdict=f2_testbench.bbsigdict_802_11n_random_QAM16_OFDM
        self.channeldict=f2_testbench.channeldict_802_11n_C
        self.DEBUG= False
        if len(arg)>=1:
            parent=arg[0]
            print(arg[1])
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        self.init()
    
    def init(self):
        #this sets the dependent variables
        self.rxmodels=[]
        [ self.rxmodels.append('py') for i in range(self.Rxantennas) ];
        self.Disableuser=[]
        [ self.Disableuser.append(False) for i in range(self.Users) ]              #Disable data transmission for cerrtain users
        self.Rxantennalocations=np.arange(self.Rxantennas)*0.3

        self.dut=f2_chip(self)
        self.dut.bbsigdict=self.bbsigdict

        #self.dut.Users=self.Users
        #self.dut.Rs=self.Rs
        self.dut.Txbits=self.Txbits
        self.dut.Txpower=0
        self.dut.init()

        #TX_signal_generator
        #Signal generator model here
        self.signal_gen_tx=f2_signal_gen(self)
        self.signal_gen_tx.Txantennas=1 #All the antennas have the same signal
        self.signal_gen_tx.Txpower=0
        self.signal_gen_tx.Users=self.Users 
        self.signal_gen_tx.bbsigdict=self.bbsigdict
        self.signal_gen_tx.Digital='True'
        self.signal_gen_tx.Rs=self.Rs_dsp
        self.signal_gen_tx.Bits=13

        # Matrix of [users,time,antennas] 
        #We could add an IO to dut.
        #Cant connect here. Formats incompatible
        #self.dut.tx_dsp.iptr_A=self.signal_gen_tx._Z

        #RX_signal_generator
        #Signal generator model here
        self.signal_gen_rx=f2_signal_gen()
        self.signal_gen_rx.Disableuser=self.Disableuser
        self.signal_gen_rx.bbsigdict=self.bbsigdict
        self.signal_gen_rx.Txantennas=1 #Only one transmitter antenna for modeling 
        self.signal_gen_rx.Txpower=self.Txpower
        self.signal_gen_rx.Users=self.Users
        self.signal_gen_rx.bbsigdict=self.bbsigdict
        self.signal_gen_rx.Rs=self.Rs

        #Add the channel between 
        self.channel=f2_channel(self)

        #Make this as an array of pointers
        self.channel.iptr_A=self.signal_gen_rx._Z

        #Connections must be propagated top down.
        #Connect rxs to channel 
        for i in range(self.Rxantennas):
            #Define connections by lower levels of hierarchies.
            self.channel._Z.Value[i]=self.dut.iptr_A.Value[i] 

    def run_tx(self):
        # Signals and channel inited in poth cases to have valid and defined inputs
        self.signal_gen_tx.init() # this must be re-inited, because output 
        self.signal_gen_rx.init()
        self.signal_gen_rx.set_transmit_power() # is not compatible with input of tx_dsp
        self.channel.run()
        self.dut.run_rx_analog()
        for i in range(self.Users):
            #Drive signal to DUT
            self.dut.dsp._io_lanes_rx[0].data[i].udata.Value=self.signal_gen_tx._Z.Value[i,:,0].reshape(-1,1)
        self.dut.run_tx_dsp()
        self.analyze_tx_dsp()

    def run_rx(self):
        # Signals and channel inited in poth cases to have valid defined inputs
        self.signal_gen_tx.init() # this must be re-inited, because output 
        self.signal_gen_rx.init()
        self.signal_gen_rx.set_transmit_power()
        self.channel.run()
        self.dut.run_rx_analog()
        for i in range(self.Users):
            #Drive signal to DUT
            self.dut.dsp._io_lanes_rx[0].data[i].udata.Value=self.signal_gen_tx._Z.Value[i,:,0].reshape(-1,1)
        self.dut.run_rx_dsp()
        self.analyze_rx_dsp()


    def analyze_rx_dsp(self):
        timex = np.array(range(len(self.signal_gen_rx._Z.Value[0,:,0])))
        
        spectrumfloor=-40
        spectrumfloorideal=-75

        #Plot signal generator outputs
        ymax=np.amax(np.amax(np.amax(np.absolute(np.real(self.signal_gen_rx._Z.Value)))))
        for k in range(self.Users):
            argdict={'timex'   :timex,
                     'sigin'   :np.real(self.signal_gen_rx._Z.Value[k,timex,0]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"Tx, User=%i" %(k),
                     'printstr':"%s/F2_system_Tx_antennas_Rs_%i_k=%i.eps" %(self.picpath, self.Rs, k)}
            self.oscilloscope(argdict)

            #Spectrum
        for i in range(self.Users):
            argdict={'sigin':self.signal_gen_rx._Z.Value[i,:,0],
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Tx, User:%i" %(i),
                     'printstr':"%s/F2_system_Tx_antennas_Spectrum_Rs_%i_k=%i.eps" %(self.picpath, self.Rs, i)} 
            self.spectrum_analyzer(**argdict)

        #Plot Rx inputs
        ymax=0
        for i in range(self.Rxantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.dut.rx[i].iptr_A.Value))))])

        for i in range(len(self.dut.rx)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.dut.rx[i].iptr_A.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr' : "Rx, Ant=%i" %(i), 
                     'printstr':"%s/F2_system_Antenna_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.oscilloscope(argdict)


            #Spectrum
        for i in range(self.Rxantennas):
            argdict={'sigin':self.dut.rx[i].iptr_A.Value,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Rx, Ant=%i" %(i), 
                     'printstr':"%s/F2_system_Antenna_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

        #Plot the rx output signals
        ymax=0
        for i in range(self.Rxantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.dut.rx[i]._Z.Value))))])
        for i in range(len(self.dut.rx)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.dut.rx[i]._Z.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"Rx, rxm=%s, Ant=%i" %(self.dut.rx[i].model,i),
                     'printstr':"%s/F2_system_Rx_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i) }
            self.oscilloscope(argdict)


            #Spectrum
        for i in range(self.Rxantennas):
            argdict={'sigin':self.dut.rx[i]._Z.Value,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Rx, Rx=%i" %(i), 
                     'printstr':"%s/F2_system_Rx_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

        #Plot the Adc output signals
        ymax=0
        for i in range(self.Rxantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.dut.adc[i]._Z.Value))))])

        for i in range(len(self.dut.adc)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.dut.adc[i]._Z.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"ADC, adcm=%s, Ant=%i" %(self.dut.adc[i].model,i),
                     'printstr': "%s/F2_system_ADC_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)} 
            self.oscilloscope(argdict)

            argdict={'sigin':self.dut.adc[i]._Z.Value,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "ADC, Rx=%i" %(i), 
                     'printstr':"%s/F2_system_ADC_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)


        ##Plot dsp and serdes outputs
        ymax=0
        k=0
        for i in range(self.Rxantennas):
                ymax=np.amax([ymax,np.amax(np.absolute(np.real(self.dut.dsp._io_lanes_tx[0].data[i].udata.Value)))])

        ##Plot the DSP output signals
        timex = np.array(range(256,len(self.dut.dsp._io_lanes_tx[0].data[0].udata.Value[:])))
        for i in range(4):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.dut.dsp._io_lanes_tx[0].data[i].udata.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"DSP, dspmodel=%s, Ant=%i" %(self.dut.dsp.model,i),
                     'printstr': "%s/F2_system_DSP_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)} 
            self.oscilloscope(argdict)

            argdict={'sigin':self.dut.dsp._io_lanes_tx[0].data[i].udata.Value[timex],
                     'ymax'    :3, 
                     'ymin'    :spectrumfloorideal,
                     'Rs'      :self.Rs_dsp,
                     'nperseg' :1024, 
                     'tstr'    : "DSP, Rx=%i" %(i), 
                     'printstr':"%s/F2_system_DSP_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

            #argdict={'timex'   :timex, 
            #         'sigin'   :np.real(self.dut.serdes._Z.Value[i].Value[timex]),
            #         'ymax'    :1.1*ymax,
            #         'ymin'    :-1.1*ymax, 
            #         'tstr'    : "Serdes, %s, Ant=%i" %(self.dut.serdes.model,i),
            #         'printstr':"%s/F2_system_serdes_Rs_%i_m=%i.eps" %(self.picpath, self.Rs,i)} 
            #self.oscilloscope(argdict)

            #argdict={'sigin'   :self.dut.serdes._Z.Value[i].Value,
            #         'ymax'    :3, 
            #         'ymin'    :spectrumfloorideal,
            #         'Rs'      :self.Rs_dsp,
            #         'nperseg' :1024, 
            #         'tstr'    : "Serdes, Rx=%i" %(i), 
            #         'printstr':"%s/F2_system_serdes_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            #self.spectrum_analyzer(**argdict)

    def analyze_tx_dsp(self):
        timex = np.array(range(0,200))
        
        spectrumfloor=-40
        spectrumfloorideal=-75

        #Plot signal generator outputs
        ymax=np.amax(np.amax(np.amax(np.absolute(np.real(self.signal_gen_tx._Z.Value)))))
        for k in range(self.Users):
            argdict={'timex'   :timex,
                     'sigin'   :np.real(self.signal_gen_tx._Z.Value[k,timex,0]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"Tx_signal gen, User=%i" %(k),
                     'printstr':"%s/F2_system_Tx_input_Rs_%i_k=%i.eps" %(self.picpath, self.Rs_dsp, k)}
            self.oscilloscope(argdict)

        #Spectrum
        timex = np.array(range(0,1023))
        for i in range(self.Users):
            argdict={'sigin':self.signal_gen_tx._Z.Value[i,:,0],
                     'Rs'   :self.Rs_dsp,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Tx signal gen, User:%i" %(i),
                     'printstr':"%s/F2_system_Tx_input_Spectrum_Rs_%i_k=%i.eps" %(self.picpath, self.Rs_dsp, i)} 
            self.spectrum_analyzer(**argdict)

        #Plot the tx output signals
        #timex = np.array(range(15000,len(self.dut.tx_dacs[0]._Z.Value)))
        #timex = np.array(range(17000,19000))
        timex = np.array(range(1024,len(self.dut.tx_dacs[0]._Z.Value)-1024))
        ymax=0
        for i in range(self.Txantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.dut.tx_dacs[i]._Z.Value))))])
        #ymax=reduce((lambda x,y:  np.amax([np.amax(np.absolute(np.real(y._Z.Value))),x])),self.dut.tx_dacs)
        for i in range(len(self.dut.tx_dacs)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.dut.tx_dacs[i]._Z.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"Tx dac, tx model=%s, Ant=%i" %(self.dut.tx_dacs[i].model,i),
                     'printstr':"%s/F2_system_Tx_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i) }
            self.oscilloscope(argdict)

        #timex = np.array(range(0,len(self.dut.tx_dacs[0]._Z.Value)))
        timex = np.array(range(1024,len(self.dut.tx_dacs[0]._Z.Value)-1024))
            #Spectrum
        for i in range(self.Txantennas):
            argdict={'sigin':self.dut.tx_dacs[i]._Z.Value[timex],
                     'Rs'   :self.Rs,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Tx dac, Tx=%i" %(i), 
                     'printstr':"%s/F2_system_Tx_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

