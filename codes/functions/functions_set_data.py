import numpy as np
from pyts.image import GramianAngularField

def Set_Data_Segmentation(Sample): 
    Segment_Set = []

    for i in range(0,int(len(Sample)/3)): # 816 = 16*51
        s = i * 3 
        Segment_Set_Temp = []

        for j in range(0,10):     # 10 = 5 seconds / 0.5 seconds
            W = j*50             # 0.5 Seconds
            Segment = np.vstack((Sample[s+0][W:W+50], Sample[s+1][W:W+50], Sample[s+2][W:W+50]))
            Segment_Set_Temp.append(Segment) # 10, 3, 50

        Segment_Set.append(Segment_Set_Temp)
        
    return(Segment_Set)

def Set_Data_GAF(Segment_Set):
    Image_Size = 30 # 경험에 의한 숫자 결정
    GASF_Setting = GramianAngularField(image_size=Image_Size, sample_range=(-1, 1), method='summation') 
    # sample_range=(-1, 1) -> 벡터의 값이 -1~1, 일반화 // method='summation' -> method에 두가지 있는데 뭘 쓰든 별로 상관 없음
    GAF_Set = []

    # Segment_Set: 816, 10, 3, 50
    for i in range(0,len(Segment_Set)): #816
        GAF_Seg_Set = []

        for j in range(0,len(Segment_Set[0])):#10
            GAF = GASF_Setting.fit_transform(Segment_Set[i][j]) #(3, 50) -> (3, 30, 30)
            GAF_Seg_Set.append(GAF) #(10, 3, 30, 30)

        GAF_Set.append(GAF_Seg_Set)
        
    return (GAF_Set) #(816, 10, 3, 30, 30)