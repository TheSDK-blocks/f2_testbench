#from joblib import Parallel, delayed
import numpy as np
import matplotlib as mpl 
mpl.use('Agg') #To enble plotting without X
import matplotlib.pyplot as plt
from thesdk import *
import multiprocessing
from joblib import Parallel, delayed
from f2_chip import *
from signal_generator_802_11n import *
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
        self.rxmodels=[]
        #Signals should be in form s(user,time,Txantenna)
        self.Txantennas=1                       #All the antennas process the same data
        self.Txpower=30                         #Output power per antenna in dBm
        self.Rxantennas=4
        self.Users=2
        self.Disableuser=[]
        self.Nbits=10  #ADC bits
        self.Txbits=9  #DAC bits
        self.Channeldir='Uplink'
        self.CPUBFmode = 'ZF';                  #['BF' | 'ZF'] Default to beam forming 
        self.DSPmode = 'cpu';                   #['cpu' | 'local'] bamforming can be local 
        self.dsp_decimator_model='py'
        self.dsp_decimator_scales=[1,1,1,1]
        self.noisetemp=290
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
        self.dut=f2_chip(self)
        #self.dut.bbsigdict=self.bbsigdict_sinusoid
        #self.dut.bbsigdict=self.bbsigdict_sinusoid3
        self.dut.bbsigdict=self.bbsigdict

        #self.dut.bbsigdict=f2s.f2_chip.bbsigdict_ofdm_sinusoid3
        #self.dut.bbsigdict=f2s.f2_chip.bbsigdict_randombitstream_QAM4_OFDM
        #self.dut.bbsigdict=f2s.f2_chip.bbsigdict_802_11n_random_QAM16_OFDM

        self.dut.Users=self.Users
        self.dut.Rs=self.Rs
        self.dut.Txbits=self.Txbits
        self.dut.Txpower=0
        self.dut.define_fader2()
        self.dut.tx_dsp.model='sv'
        self.dut.init()
        self.dut.define_fader2()

        #TX_signal_generator
        #Signal generator model here
        self.signal_gen_tx=f2_signal_gen(self)
        self.signal_gen_tx.Txantennas=self.Txantennas 
        self.signal_gen_tx.Txpower=0
        self.signal_gen_tx.Users=self.Users 
        self.signal_gen_tx.bbsigdict=self.bbsigdict
        self.signal_gen_tx.Digital='True'
        self.signal_gen_tx.Rs=self.Rs_dsp
        self.signal_gen_tx.Bits=self.Txbits

        # Matrix of [1,time,users] 
        #We could add an IO to dut.
        self.dut.tx_dsp.iptr_A=self.signal_gen_tx._Z

        #RX_signal_generator
        #Signal generator model here
        self.signal_gen_rx=f2_signal_gen(self)
        self.signal_gen_rx.bbsigdict=self.bbsigdict
        self.signal_gen_rx.Txantennas=self.Rxantennas 
        self.signal_gen_rx.Txpower=self.Txpower
        self.signal_gen_rx.Users=self.Users 
        self.signal_gen_rx.bbsigdict=self.bbsigdict
        self.signal_gen_rx.Rs=self.Rs

        #The mached filters for the symbol synchronization
        #These are considered reconfigurable
        self.Hstf=np.conj(self.signal_gen_rx.sg802_11n._PLPCseq_short[0:64])
        self.dut.Hstf=self.Hstf[::-1]
        self.Hltf=np.conj(self.signal_gen_rx.sg802_11n._PLPCseq_long[0:16])
        self.dut.Hltf=self.Hltf[::-1]
        self.signal_gen_rx.set_transmit_power() #default 30dBm

        #Add the channel between 
        self.channel=f2_channel(self)
        print(len(self.channel._Z.Value))

        #Make this as an array of pointers
        self.channel.iptr_A=self.signal_gen_rx._Z

        #Connect rxs to channel 
        for i in range(self.Rxantennas):
            self.dut.rx[i].iptr_A=self.channel._Z.Value[i]
    
    def run_tx(self):
        self.signal_gen_tx.init()
        self.dut.run_tx_dsp()

    def run_rx(self):
        self.signal_gen_rx.init()
        self.channel.run()
        self.dut.run_rx_dsp()

    def analyze_simple(self):
        timex = np.array(range(0,10000))
        spectrumfloor=-40
        spectrumfloorideal=-75
        #Plot signal generator outputs
        ymax=np.amax(np.amax(np.amax(np.absolute(np.real(self.signal_gen._Z.Value)))))
        for k in range(self.Users):
            argdict={'I'       :np.real(self.signal_gen._qam_reference[k,:]), 
                     'Q'       :np.imag(self.signal_gen._qam_reference[k,:]),
                     'ymax'    :1.1*np.max(np.abs(np.real(self.signal_gen._qam_reference[k,:]))),
                     'ymin'    :-1.1*np.max(np.abs(np.imag(self.signal_gen._qam_reference[k,:]))), 
                     'tstr'    : "QAM data, Usr=%i" %(k),
                     'printstr':"%s/F2_system_QAM_reference_Rs_%i_k=%i.eps" %(self.picpath, self.Rs,k)}
            self.constellation(argdict)

            argdict={'timex'   :timex,
                     'sigin'   :np.real(self.signal_gen._Z.Value[k,timex,0]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"Tx, User=%i" %(k),
                     'printstr':"%s/F2_system_Tx_antennas_Rs_%i_k=%i.eps" %(self.picpath, self.Rs, k)}
            self.oscilloscope(argdict)

            #Spectrum
        for i in range(self.Users):
            argdict={'sigin':self.signal_gen._Z.Value[i,:,0],
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Tx, User:%i" %(i),
                     'printstr':"%s/F2_system_Tx_antennas_Spectrum_Rs_%i_k=%i.eps" %(self.picpath, self.Rs, i)} 
            self.spectrum_analyzer(**argdict)

        #Plot Rx inputs
        ymax=0
        for i in range(self.Rxantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.rx[i].iptr_A.Value))))])

        for i in range(len(self.rx)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.rx[i].iptr_A.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr' : "Rx, Ant=%i" %(i), 
                     'printstr':"%s/F2_system_Antenna_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.oscilloscope(argdict)


            #Spectrum
        for i in range(self.Rxantennas):
            argdict={'sigin':self.rx[i].iptr_A.Value,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Rx, Ant=%i" %(i), 
                     'printstr':"%s/F2_system_Antenna_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

        #Plot the rx output signals
        ymax=0
        for i in range(self.Rxantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.rx[i]._Z.Value))))])
        for i in range(len(self.rx)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.rx[i]._Z.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"Rx, rxm=%s, Ant=%i" %(self.rx[i].model,i),
                     'printstr':"%s/F2_system_Rx_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i) }
            self.oscilloscope(argdict)


            #Spectrum
        for i in range(self.Rxantennas):
            argdict={'sigin':self.rx[i]._Z.Value,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Rx, Rx=%i" %(i), 
                     'printstr':"%s/F2_system_Rx_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

        #Plot the Adc output signals
        ymax=0
        for i in range(self.Rxantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.adc[i]._Z.Value))))])

        for i in range(len(self.adc)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.adc[i]._Z.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"ADC, adcm=%s, Ant=%i" %(self.adc[i].model,i),
                     'printstr': "%s/F2_system_ADC_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)} 
            self.oscilloscope(argdict)

            argdict={'sigin':self.adc[i]._Z.Value,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "ADC, Rx=%i" %(i), 
                     'printstr':"%s/F2_system_ADC_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

        #Plot dsp and serdes
        ymax=0
        ymaxfsyncl=0
        ymaxfsyncs=0
        syncrange=np.array(range(0,700))
        for i in range(self.Rxantennas):
            for k in range(self.Users):
                ymax=np.amax([ymax,np.amax(np.absolute(np.real(self.serdes._Z.Value[timex,k])))])

        for i in range(self.Rxantennas):
            ymaxfsyncs=np.amax([ymaxfsyncs,np.amax(np.absolute(self.dsp[i]._Frame_sync_short.Value[syncrange]))])
            ymaxfsyncl=np.amax([ymaxfsyncl,np.amax(np.absolute(self.dsp[i]._Frame_sync_long.Value[syncrange]))])

        for i in range(self.Rxantennas):
            argdict={'timex'   :syncrange,
                     'sigin'   :np.real(self.dsp[i]._Frame_sync_short.Value[syncrange]),
                     'ymax'    :1.1*ymaxfsyncs,
                     'ymin'    :0, 
                     'tstr'    : "DSP Framesync short, Ant=%i, model %s" %(i, self.dsp[i].model),
                     'printstr':"%s/F2_system_DSP_Fsync_short_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.oscilloscope(argdict)

            argdict={'timex'   :syncrange,
                     'sigin'   :self.dsp[i]._Frame_sync_long.Value[syncrange],
                     'ymax'    :1.1*ymaxfsyncl,
                     'ymin'    :0, 
                     'tstr'    : "DSP Framesync long, Ant=%i, model %s" %(i, self.dsp[i].model),
                     'printstr':"%s/F2_system_DSP_Fsync_long_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.oscilloscope(argdict)

        for i in range(self.Rxantennas):
            for k in range(self.Users):
                l=np.amin([self.signal_gen._qam_reference.shape[1], self.dsp[i]._symbols.Value[k].Value.shape[0]])
                EVM=self.evm_calculator({'ref':self.signal_gen._qam_reference[k,0:l],'signal':self.dsp[i]._symbols.Value[k].Value[0:l]})
                l=np.amin([self.signal_gen._bitstream_reference.shape[1], self.dsp[i]._bitstream.Value[k].Value.shape[0]])
                BER=self.ber_calculator({'ref':self.signal_gen._bitstream_reference[k,0:l],'signal':self.dsp[i]._bitstream.Value[k].Value[0:l]})
                argdict={'I'       :np.real(self.dsp[i]._symbols.Value[k].Value), 
                         'Q'       :np.imag(self.dsp[i]._symbols.Value[k].Value),
                         'ymax'    :1.1*ymax,
                         'ymin'    :-1.1*ymax, 
                         'tstr'    : "DSP, Ant=%i, Usr=%i, %s, EVM=%0.2f dB, BER= %0.3g" %(i, k, self.dsp[i].model, EVM,BER),
                         'printstr':"%s/F2_system_DSP_Rs_%i_m=%i_k=%i.eps" %(self.picpath, self.Rs, i,k)}
                self.constellation(argdict)


        for k in range(self.Users):
            l=np.amin([self.signal_gen._qam_reference.shape[1], self.postproc._symbols.Value.shape[0]])
            EVM=self.evm_calculator({'ref':self.signal_gen._qam_reference[k,0:l],'signal':self.postproc._symbols.Value[0:l,k]})
            l=np.amin([self.signal_gen._bitstream_reference.shape[1], self.postproc._bitstream.Value.shape[0]])
            BER=self.ber_calculator({'ref':self.signal_gen._bitstream_reference[k,0:l],'signal':self.postproc._bitstream.Value[0:l,k]})
            argdict={'I'       :np.real(self.postproc._symbols.Value[:,k]), 
                    'Q'       :np.imag(self.postproc._symbols.Value[:,k]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    : "Postproc, Usr=%i, %s, EVM=%0.2f dB, BER= %0.3g" %(k, self.postproc.model, EVM,BER),
                     'printstr':"%s/F2_system_Postproc_Rs_%i_k=%i.eps" %(self.picpath, self.Rs,k)}
            self.constellation(argdict)


            argdict={'timex'   :timex, 
                     'sigin'   :self.serdes._Z.Value[timex,k],
                     'ymax'    :1.1*ymax,
                     'ymin'    :0, 
                     'tstr'    : "Serdes, %s, Usr=%i" %(self.serdes.model,k),
                     'printstr':"%s/F2_system_serdes_Rs_%i_k=%i.eps" %(self.picpath, self.Rs,k)} 
            self.oscilloscope(argdict)

    def analyze_rx_dsp(self):
        timex = np.array(range(0,10000))
        
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
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.rx[i].iptr_A.Value))))])

        for i in range(len(self.rx)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.rx[i].iptr_A.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr' : "Rx, Ant=%i" %(i), 
                     'printstr':"%s/F2_system_Antenna_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.oscilloscope(argdict)


            #Spectrum
        for i in range(self.Rxantennas):
            argdict={'sigin':self.rx[i].iptr_A.Value,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Rx, Ant=%i" %(i), 
                     'printstr':"%s/F2_system_Antenna_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

        #Plot the rx output signals
        ymax=0
        for i in range(self.Rxantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.rx[i]._Z.Value))))])
        for i in range(len(self.rx)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.rx[i]._Z.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"Rx, rxm=%s, Ant=%i" %(self.rx[i].model,i),
                     'printstr':"%s/F2_system_Rx_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i) }
            self.oscilloscope(argdict)


            #Spectrum
        for i in range(self.Rxantennas):
            argdict={'sigin':self.rx[i]._Z.Value,
                     'ymax':3, 
                     'ymin':spectrumfloorideal,
                     'nperseg':1024, 
                     'tstr' : "Rx, Rx=%i" %(i), 
                     'printstr':"%s/F2_system_Rx_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

        #Plot the Adc output signals
        ymax=0
        for i in range(self.Rxantennas):
            ymax=np.amax([ymax, np.amax(np.amax(np.absolute(np.real(self.adc[i]._Z.Value))))])

        for i in range(len(self.adc)):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.adc[i]._Z.Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"ADC, adcm=%s, Ant=%i" %(self.adc[i].model,i),
                     'printstr': "%s/F2_system_ADC_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)} 
            self.oscilloscope(argdict)

            argdict={'sigin':self.adc[i]._Z.Value,
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
                ymax=np.amax([ymax,np.amax(np.absolute(np.real(self.serdes._Z.Value[i].Value)))])

        ##Plot the DSP output signals
        for i in range(4):
            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.rx_dsp._decimated.Value[i].Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    :"DSP, dspmodel=%s, Ant=%i" %(self.rx_dsp.model,i),
                     'printstr': "%s/F2_system_DSP_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)} 
            self.oscilloscope(argdict)

            argdict={'sigin':self.rx_dsp._decimated.Value[i].Value  ,
                     'ymax'    :3, 
                     'ymin'    :spectrumfloorideal,
                     'Rs'      :self.Rs_dsp,
                     'nperseg' :1024, 
                     'tstr'    : "DSP, Rx=%i" %(i), 
                     'printstr':"%s/F2_system_DSP_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

            argdict={'timex'   :timex, 
                     'sigin'   :np.real(self.serdes._Z.Value[i].Value[timex]),
                     'ymax'    :1.1*ymax,
                     'ymin'    :-1.1*ymax, 
                     'tstr'    : "Serdes, %s, Ant=%i" %(self.serdes.model,i),
                     'printstr':"%s/F2_system_serdes_Rs_%i_m=%i.eps" %(self.picpath, self.Rs,i)} 
            self.oscilloscope(argdict)

            argdict={'sigin'   :self.serdes._Z.Value[i].Value,
                     'ymax'    :3, 
                     'ymin'    :spectrumfloorideal,
                     'Rs'      :self.Rs_dsp,
                     'nperseg' :1024, 
                     'tstr'    : "Serdes, Rx=%i" %(i), 
                     'printstr':"%s/F2_system_serdes_Spectrum_Rs_%i_m=%i.eps" %(self.picpath, self.Rs, i)}
            self.spectrum_analyzer(**argdict)

