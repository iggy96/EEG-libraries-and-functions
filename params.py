from df_lib import np

ant = dict(
    fs=500,                            # Hz - sampling rate of the amplifier
    dataCols=12,                       # Number of columns in the EEG
    nEEG=8,                            # Number of EEG Channels
    trigCol=-1,                        # Position of Trigger Channel
    uvScale=1000000,                   # antNeuro displays numbers in volts
    dt=np.float64,                     # data type to read in
    eegChans=np.array([0, 1, 2]),      # index of Fz, Cz, Pz within raw data array
    eegChanNames = ['Fz', 'Cz', 'Pz'],
    eogChans = np.array([5]),  # FPZ for now    
    afSS=0                             # Start sample of the adaptive filter (transient)
)


gtec = dict(
    fs=500,                            # Hz - sampling rate of the amplifier
    dataCols=14,                       # Number of columns in the EEG
    nEEG=8,                            # Number of EEG Channels
    trigCol=-1,                        # Position of Trigger Channel
    uvScale=1,                         # gtec already displays numbers in microvolts
    dt=np.float32,                     # data type to read in
    eegChans=np.array([0, 1, 3]),      # index of Fz, Cz, Pz within raw data array
    eegChanNames=['Fz', 'Cz', 'Pz'],
    eogChans=np.array([2,4,5,6,7]),    # index of EOG (P3 P4 P07 P08 OZ) channels within raw data 
    afSS=3000                          # Start sample of the adaptive filter (transient)
)
