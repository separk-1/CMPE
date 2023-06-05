import scipy.signal as signal
import numpy as np

def Set_Filter_BL(Data):
    Order = 5
    Cutoff = 10
    DelayNumber = 100
    SR = 128 #100?

    # Parameters #
    NYQ = 0.5 * SR
    Normal_Cutoff = Cutoff / NYQ
    B, A = signal.butter(Order, Normal_Cutoff, btype='low', analog=False)
    
    
    # Data #
    BL_Data = np.zeros(Data.shape)
    PacketNumber = len(Data)
    Sub_Data = np.zeros(PacketNumber + DelayNumber,)
    Sub_Data[0:DelayNumber] = Data[0:DelayNumber]
    Sub_Data[DelayNumber:PacketNumber + DelayNumber] = Data


    # Filtering #
    BL_Sub_Data = signal.lfilter(B, A, Sub_Data)
    BL_Data = BL_Sub_Data[DelayNumber:PacketNumber + DelayNumber]

    
    return BL_Data