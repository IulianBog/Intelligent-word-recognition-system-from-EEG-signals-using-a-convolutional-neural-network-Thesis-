#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import pathlib
import sys
import ntpath
import glob, os
import random



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense,Conv2D,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split

import pandas
# pandas.__version__
get_ipython().run_line_magic('matplotlib', 'qt')


# In[10]:


#applying ICA method


path=r'C:\Users\Iulian\Desktop\Facultate\Licenta\Baza_date_semnale_raw' #path to the data_base
fisiere=os.listdir(path)


drop_ch = (['M1', 'M2','VEO', 'HEO', 'EKG', 'EMG', 'Trigger'])
ica = mne.preprocessing.ICA(n_components=57, random_state=0)

covariatie = np.zeros((len(fisiere),62,62))

eeg_filtrat=[];
eeg_filtrat=np.zeros((len(fisiere),62,62))

for i,j in zip(fisiere,range(len(fisiere))):
    aux=os.path.join(path,i)
    if(aux[-4:len(aux)]=='.fif'):
        eeg=mne.io.read_raw_fif(aux,preload=True)
        eeg.drop_channels(drop_ch)
        ica.fit(eeg.copy().filter(0, 50))
        bad_idx, scores = ica.find_bads_eog(eeg, 'FP1', threshold=1.5)
        
        eeg=ica.apply(eeg.copy(), exclude=bad_idx)
        eeg_filtrat=np.array(eeg[:][0])
        covariatie[j,:,:]=np.cov(eeg_filtrat)

np.save("eeg_filtrat.npy",covariatie)


# In[7]:


path=r'C:\Users\Iulian\Desktop\Facultate\Licenta\Baza_date_semnale_raw'
fisiere=os.listdir(path)

eeg_filtrat=np.load("eeg_filtrat.npy")
covariatie = np.zeros((len(fisiere),62,62))

medie = np.mean(eeg_filtrat,axis=(1,2))

for i in range(len(medie)):
    std = np.std(eeg_filtrat[i])
    eeg_filtrat[i]=np.subtract(eeg_filtrat[i],medie[i])/std


for i,j in zip(fisiere,range(len(fisiere))):
    covariatie[j,:,:]=np.cov(eeg_filtrat[j])
print(covariatie)
np.save("eeg_filtrat_standardizat.npy",covariatie)


# In[8]:


path=r'C:\Users\Iulian\Desktop\Facultate\Licenta\Baza_date_semnale_raw'


def file(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path,file)):
            yield file
i=0
cnt = np.zeros(993)
vector_bilabial= np.zeros(993)
vector_nasal= np.zeros(993)
vector_cv= np.zeros(993)
vector_uw= np.zeros(993)
vector_iy= np.zeros(993)

tag_0= []
tag_1= []
tag_2= []
tag_3= []
tag_4= []
tag_5= []
tag_6= []
tag_7= []
tag_8= []
tag_9= []
tag_10= []


tag_0_testare= []
tag_1_testare= []
tag_2_testare= []
tag_3_testare= []
tag_4_testare= []
tag_5_testare= []
tag_6_testare= []
tag_7_testare= []
tag_8_testare= []
tag_9_testare= []
tag_10_testare= []


for file in os.listdir(path):
                    string = r'C:\Users\Iulian\Desktop\Facultate\Licenta\Baza_date_semnale_raw' +  file
                    raw_string = r"{}".format(string)
                    if(raw_string[string.index('tag')+4] != '0'):
                        cnt[i] = raw_string[string.index('tag')+3]
                    else:
                        cnt[i] = '10'
                        
                    if(cnt[i]==0):
                        tag_0.append(i)
                    if(cnt[i]==1):
                        tag_1.append(i)
                    if(cnt[i]==2):
                        tag_2.append(i)
                    if(cnt[i]==3):
                        tag_3.append(i)
                    if(cnt[i]==4):
                        tag_4.append(i)
                    if(cnt[i]==5):
                        tag_5.append(i)
                    if(cnt[i]==6):
                        tag_6.append(i)
                    if(cnt[i]==7):
                        tag_7.append(i)
                    if(cnt[i]==8):
                        tag_8.append(i)
                    if(cnt[i]==9):
                        tag_9.append(i)
                    if(cnt[i]==10):
                        tag_10.append(i)
                        
                    if(cnt[i]==2 or cnt[i]==5 or cnt[i]==7 or cnt[i]==8):
                        vector_bilabial[i]=1
                    if(cnt[i]==5 or cnt[i]==6 or cnt[i]==9 or cnt[i]==10):
                        vector_nasal[i]=1
                    if(cnt[i]==2 or cnt[i]==3 or cnt[i]==4 or cnt[i]==5 or cnt[i]==6):
                        vector_cv[i]=1
#                         1- consoane
#                         2- vocale
                    if(cnt[i]==1):
                        vector_uw[i]=1
                    if(cnt[i]==0):
                        vector_iy[1]=1
                    i=i+1


np.save("tag_0.npy",tag_0)
np.save("tag_1.npy",tag_1)
np.save("tag_2.npy",tag_2)
np.save("tag_3.npy",tag_3)
np.save("tag_4.npy",tag_4)
np.save("tag_5.npy",tag_5)
np.save("tag_6.npy",tag_6)
np.save("tag_7.npy",tag_7)
np.save("tag_8.npy",tag_8)
np.save("tag_9.npy",tag_9)
np.save("tag_10.npy",tag_10)



np.save("vector_bilabial.npy",vector_bilabial)
np.save("vector_nasal.npy",vector_nasal)
np.save("vector_cv.npy",vector_cv)

# # print(cnt)

# # print (vector_bilabial)                    
# # print (vector_nasal)
# # print (vector_cv)

    # 1- true
    # 0- false

# # eeg = mne.io.read_raw_fif(r'C:\Users\Iulian\Desktop\Facultate\Licenta\Baza_date_semnale_raw\imagined_speech_MM05_20_tag6.raw.fif', preload = True)


# In[34]:


tag_antrenare=[]
tag_test=[]

tag_0_test= []
tag_1_test= []
tag_2_test= []
tag_3_test= []
tag_4_test= []
tag_5_test= []
tag_6_test= []
tag_7_test= []
tag_8_test= []
tag_9_test= []
tag_10_test= []
tag_aux=[]

tag_0=np.load("tag_0.npy")
tag_1=np.load("tag_1.npy")
tag_2=np.load("tag_2.npy")
tag_3=np.load("tag_3.npy")
tag_4=np.load("tag_4.npy")
tag_5=np.load("tag_5.npy")
tag_6=np.load("tag_6.npy")
tag_7=np.load("tag_7.npy")
tag_8=np.load("tag_8.npy")
tag_9=np.load("tag_9.npy")
tag_10=np.load("tag_10.npy")


tag_aux=tag_0
np.random.shuffle(tag_aux)

tag_0_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_0 = tag_aux[0:int(0.8 * len(tag_aux))]



for i in range (0,len(tag_0)):
    tag_antrenare.append(tag_0[i])
for i in range (0,len(tag_0_test)):
    tag_test.append(tag_0_test[i])
    
# # ________________________________________________


tag_aux=tag_1
np.random.shuffle(tag_aux)

tag_1_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_1 = tag_aux[0:int(0.8 * len(tag_aux))]


for i in range (0,len(tag_1)):
    tag_antrenare.append(tag_1[i])
for i in range (0,len(tag_1_test)):
    tag_test.append(tag_1_test[i])


# ________________________________________________


tag_aux=tag_2
np.random.shuffle(tag_aux)

tag_2_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_2 = tag_aux[0:int(0.8 * len(tag_aux))]


for i in range (0,len(tag_2)):
    tag_antrenare.append(tag_2[i])
for i in range (0,len(tag_2_test)):
    tag_test.append(tag_2_test[i])
    
    
# ________________________________________________
    
    
tag_aux=tag_3
np.random.shuffle(tag_aux)

tag_3_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_3 = tag_aux[0:int(0.8 * len(tag_aux))]

    
for i in range (0,len(tag_3)):
    tag_antrenare.append(tag_3[i])
for i in range (0,len(tag_3_test)):
    tag_test.append(tag_3_test[i])
    
# ________________________________________________
    

tag_aux=tag_4
np.random.shuffle(tag_aux)

tag_4_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_4 = tag_aux[0:int(0.8 * len(tag_aux))]


for i in range (0,len(tag_4)):
    tag_antrenare.append(tag_4[i])
for i in range (0,len(tag_4_test)):
    tag_test.append(tag_4_test[i])
    
    
# ________________________________________________


tag_aux=tag_5
np.random.shuffle(tag_aux)

tag_5_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_5 = tag_aux[0:int(0.8 * len(tag_aux))]

    
for i in range (0,len(tag_5)):
    tag_antrenare.append(tag_5[i])
for i in range (0,len(tag_5_test)):
    tag_test.append(tag_5_test[i])

    
# ________________________________________________


tag_aux=tag_6
np.random.shuffle(tag_aux)

tag_6_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_6 = tag_aux[0:int(0.8 * len(tag_aux))]

    
for i in range (0,len(tag_6)):
    tag_antrenare.append(tag_6[i])
for i in range (0,len(tag_6_test)):
    tag_test.append(tag_6_test[i])
    
# ________________________________________________


tag_aux=tag_7
np.random.shuffle(tag_aux)

tag_7_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_7 = tag_aux[0:int(0.8 * len(tag_aux))]

for i in range (0,len(tag_7)):
    tag_antrenare.append(tag_7[i])
for i in range (0,len(tag_7_test)):
    tag_test.append(tag_7_test[i])


# ________________________________________________


tag_aux=tag_8
np.random.shuffle(tag_aux)

tag_8_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_8 = tag_aux[0:int(0.8 * len(tag_aux))]

for i in range (0,len(tag_8)):
    tag_antrenare.append(tag_8[i])
for i in range (0,len(tag_8_test)):
    tag_test.append(tag_8_test[i])


# ________________________________________________


tag_aux=tag_9
np.random.shuffle(tag_aux)

tag_9_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_9 = tag_aux[0:int(0.8 * len(tag_aux))]


for i in range (0,len(tag_9)):
    tag_antrenare.append(tag_9[i])
for i in range (0,len(tag_9_test)):
    tag_test.append(tag_9_test[i])

    
# ________________________________________________


tag_aux=tag_10
np.random.shuffle(tag_aux)

tag_10_test = tag_aux[int(0.8 * len(tag_aux)):len(tag_aux)]
tag_10 = tag_aux[0:int(0.8 * len(tag_aux))]


for i in range (0,len(tag_10)):
    tag_antrenare.append(tag_10[i])
for i in range (0,len(tag_10_test)):
    tag_test.append(tag_10_test[i])

np.random.shuffle(tag_test)
np.random.shuffle(tag_antrenare)

tag_test


# In[35]:


# antrenare_bilabial

date_intrare_antrenare_bilabial=[]
data_intrare_antrenare_bilabial=np.array(date_intrare_antrenare_bilabial)
date_iesire_antrenare_bilabial=[]
date_iesire_antrenare_bilabial=np.zeros(len(tag_antrenare))

covariatie=np.load("eeg_filtrat_standardizat.npy")
date_intrare_antrenare_bilabial=covariatie[tag_antrenare]



for i in range(0,len(tag_antrenare)):
    if(cnt[tag_antrenare[i]]==2 or cnt[tag_antrenare[i]]==5 or cnt[tag_antrenare[i]]==7 or cnt[tag_antrenare[i]]==8):
        date_iesire_antrenare_bilabial[i]=1

# testare_bilabial

date_intrare_test_bilabial=[]
date_intrare_test_bilabial=np.array(date_intrare_test_bilabial)
date_iesire_test_bilabial=[]
date_iesire_test_bilabial=np.zeros(len(tag_test))

covariatie=np.load("eeg_filtrat_standardizat.npy")

date_intrare_test_bilabial= covariatie[tag_test]


for i in range(len(tag_test)):
    if(cnt[tag_test[i]]==2 or cnt[tag_test[i]]==5 or cnt[tag_test[i]]==7 or cnt[tag_test[i]]==8):
        date_iesire_test_bilabial[i]=1
        
        
        
        
# date_iesire_antrenare, date_iesire_test --- bilabial

date_iesire_antrenare_aux_bilabial=[]
date_iesire_antrenare_aux_bilabial=np.zeros((date_iesire_antrenare_bilabial.shape[0],2))

for i in range (date_iesire_antrenare_bilabial.shape[0]):
    if( date_iesire_antrenare_bilabial[i] == 1):
        date_iesire_antrenare_aux_bilabial[i]=[0,1]
    else:
        date_iesire_antrenare_aux_bilabial[i]=[1,0]

date_iesire_test_aux_bilabial=np.zeros((len(date_iesire_test_bilabial),2))
for i in range (len(date_iesire_test_bilabial)):
    if(date_iesire_test_bilabial[i] == 1):
        date_iesire_test_aux_bilabial[i]=[0,1]
    else:
        date_iesire_test_aux_bilabial[i]=[1,0]


date_iesire_antrenare_aux_bilabial


# In[42]:


# antrenare_nasal



date_intrare_antrenare_nasal=[]
data_intrare_antrenare_nasal=np.array(date_intrare_antrenare_nasal)
date_iesire_antrenare_nasal=[]
date_iesire_antrenare_nasal=np.zeros(len(tag_antrenare))

covariatie=np.load("eeg_filtrat_standardizat.npy")
date_intrare_antrenare_nasal=covariatie[tag_antrenare]



for i in range(0,len(tag_antrenare)):
    if(cnt[tag_antrenare[i]]==6 or cnt[tag_antrenare[i]]==5 or cnt[tag_antrenare[i]]==9 or cnt[tag_antrenare[i]]==10):
        date_iesire_antrenare_nasal[i]=1

# testare_nasal

date_intrare_test_nasal=[]
date_intrare_test_nasal=np.array(date_intrare_test_nasal)
date_iesire_test_nasal=[]
date_iesire_test_nasal=np.zeros(len(tag_test))

covariatie=np.load("eeg_filtrat_standardizat.npy")

date_intrare_test_nasal= covariatie[tag_test]


for i in range(len(tag_test)):
    if(cnt[tag_test[i]]==6 or cnt[tag_test[i]]==5 or cnt[tag_test[i]]==9 or cnt[tag_test[i]]==10):
        date_iesire_test_nasal[i]=1
        
# date_iesire_antrenare, date_iesire_test --- nasal

date_iesire_antrenare_aux_nasal=np.zeros((date_iesire_antrenare_nasal.shape[0],2))
for i in range (date_iesire_antrenare_nasal.shape[0]):
    if(date_iesire_antrenare_nasal[i] == 1):
        date_iesire_antrenare_aux_nasal[i]=[0,1]
    else:
        date_iesire_antrenare_aux_nasal[i]=[1,0]

date_iesire_test_aux_nasal=np.zeros((date_iesire_test_nasal.shape[0],2))
for i in range (date_iesire_test_nasal.shape[0]):
    if(date_iesire_test_nasal[i] == 1):
        date_iesire_test_aux_nasal[i]=[0,1]
    else:
        date_iesire_test_aux_nasal[i]=[1,0]


# In[12]:


# antrenare_consoana/vocala



date_intrare_antrenare_cv=[]
date_intrare_antrenare_cv=np.array(date_intrare_antrenare_cv)
date_iesire_antrenare_cv=[]
date_iesire_antrenare_cv=np.zeros(len(tag_antrenare))

covariatie=np.load("eeg_filtrat_standardizat.npy")
date_intrare_antrenare_cv=covariatie[tag_antrenare]



for i in range(0,len(tag_antrenare)):
    if(cnt[tag_antrenare[i]]==2 or cnt[tag_antrenare[i]]==3 or cnt[tag_antrenare[i]]==4 or cnt[tag_antrenare[i]]==5 or cnt[tag_antrenare[i]]==6):
        date_iesire_antrenare_cv[i]=1

# testare_consoana/vocala

date_intrare_test_cv=[]
date_intrare_test_cv=np.array(date_intrare_test_cv)
date_iesire_test_cv=[]
date_iesire_test_cv=np.zeros(len(tag_test))

covariatie=np.load("eeg_filtrat_standardizat.npy")

date_intrare_test_cv= covariatie[tag_test]


for i in range(len(tag_test)):
    if(cnt[tag_test[i]]==2 or cnt[tag_test[i]]==3 or cnt[tag_test[i]]==4 or cnt[tag_test[i]]==5 or cnt[tag_test[i]]==6):
        date_iesire_test_cv[i]=1
        
# date_iesire_antrenare, date_iesire_test --- consoana/vocala

date_iesire_antrenare_aux_cv=np.zeros((len(date_iesire_antrenare_cv),2))
for i in range (len(date_iesire_antrenare_cv)):
    if(date_iesire_antrenare_cv[i] == 1):
        date_iesire_antrenare_aux_cv[i]=[0,1]
    else:
        date_iesire_antrenare_aux_cv[i]=[1,0]

date_iesire_test_aux_cv=np.zeros((len(date_iesire_test_cv),2))
for i in range (len(date_iesire_test_cv)):
    if(date_iesire_test_cv[i] == 1):
        date_iesire_test_aux_cv[i]=[0,1]
    else:
        date_iesire_test_aux_cv[i]=[1,0]


date_iesire_antrenare_aux_cv.shape


# In[13]:


# antrenare_uw



date_intrare_antrenare_uw=[]
data_intrare_antrenare_uw=np.array(date_intrare_antrenare_uw)
date_iesire_antrenare_uw=[]
date_iesire_antrenare_uw=np.zeros(len(tag_antrenare))

covariatie=np.load("eeg_filtrat_standardizat.npy")
date_intrare_antrenare_uw=covariatie[tag_antrenare]



for i in range(0,len(tag_antrenare)):
    if(cnt[tag_antrenare[i]]==1):
        date_iesire_antrenare_uw[i]=1

# testare_uw

date_intrare_test_uw=[]
date_intrare_test_uw=np.array(date_intrare_test_uw)
date_iesire_test_uw=[]
date_iesire_test_uw=np.zeros(len(tag_test))

covariatie=np.load("eeg_filtrat_standardizat.npy")

date_intrare_test_uw= covariatie[tag_test]


for i in range(len(tag_test)):
    if(cnt[tag_test[i]]==1):
        date_iesire_test_uw[i]=1
        
# date_iesire_antrenare, date_iesire_test --- uw

date_iesire_antrenare_aux_uw=np.zeros((date_iesire_antrenare_uw.shape[0],2))
for i in range (date_iesire_antrenare_uw.shape[0]):
    if(date_iesire_antrenare_uw[i] == 1):
        date_iesire_antrenare_aux_uw[i]=[0,1]
    else:
        date_iesire_antrenare_aux_uw[i]=[1,0]

date_iesire_test_aux_uw=np.zeros((date_iesire_test_uw.shape[0],2))
for i in range (date_iesire_test_uw.shape[0]):
    if(date_iesire_test_uw[i] == 1):
        date_iesire_test_aux_uw[i]=[0,1]
    else:
        date_iesire_test_aux_uw[i]=[1,0]


date_iesire_antrenare_uw


# In[50]:


np.random.shuffle(tag_test)
np.random.shuffle(tag_antrenare)


# In[39]:


tag_test


# In[51]:


# antrenare_iy



date_intrare_antrenare_iy=[]
data_intrare_antrenare_iy=np.array(date_intrare_antrenare_iy)
date_iesire_antrenare_iy=[]
date_iesire_antrenare_iy=np.zeros(len(tag_antrenare))

covariatie=np.load("eeg_filtrat_standardizat.npy")
date_intrare_antrenare_iy=covariatie[tag_antrenare]



for i in range(0,len(tag_antrenare)):
    if(cnt[tag_antrenare[i]]==0):
        date_iesire_antrenare_iy[i]=1

# testare_iy

date_intrare_test_iy=[]
date_intrare_test_iy=np.array(date_intrare_test_iy)
date_iesire_test_iy=[]
date_iesire_test_iy=np.zeros(len(tag_test))

covariatie=np.load("eeg_filtrat_standardizat.npy")

date_intrare_test_iy = covariatie[tag_test]


for i in range(len(tag_test)):
    if(cnt[tag_test[i]]==0):
        date_iesire_test_iy[i]=1
        
# date_iesire_antrenare, date_iesire_test --- iy

date_iesire_antrenare_aux_iy=np.zeros((date_iesire_antrenare_iy.shape[0],2))
for i in range (date_iesire_antrenare_iy.shape[0]):
    if(date_iesire_antrenare_iy[i] == 1):
        date_iesire_antrenare_aux_iy[i]=[0,1]
    else:
        date_iesire_antrenare_aux_iy[i]=[1,0]

date_iesire_test_aux_iy=np.zeros((date_iesire_test_iy.shape[0],2))
for i in range (date_iesire_test_iy.shape[0]):
    if(date_iesire_test_iy[i] == 1):
        date_iesire_test_aux_iy[i]=[0,1]
    else:
        date_iesire_test_aux_iy[i]=[1,0]


date_iesire_antrenare_aux_iy.shape


# In[29]:


date_intrare_antrenare=[]
date_intrare_test=[]
date_iesire_antrenare=[]
date_iesire_test=[]

covariatie=np.load("eeg_filtrat_standardizat.npy")
vector_bilabial=np.load("vector_bilabial.npy")

shuffle= np.arange(covariatie.shape[0])
np.random.shuffle(shuffle)
covariatie= covariatie[shuffle]
vector_bilabial= vector_bilabial[shuffle]

date_intrare_antrenare, date_intrare_test, date_iesire_antrenare, date_iesire_test = train_test_split(covariatie, vector_bilabial, test_size=0.2,random_state=42,shuffle=True)

# print(shuffle)

date_intrare_antrenare.shape
# date_iesire_antrenare.shape
# print(date_intrare_antrenare)
# print(date_iesire_antrenare)


# In[52]:


# Model retea neurala

model = Sequential([
    Conv2D(filters=32,kernel_size=(3,3), input_shape=(62,62,1),activation='relu'),
    Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    Dropout(rate=0.25),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(128,activation='relu'),
    Dropout(rate=0.5),
    Dense(2, activation='softmax')
])


model.summary()


# In[53]:


model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])


# In[54]:


# date_intrare_antrenare_bilabial=date_intrare_antrenare_bilabial.reshape(-1,62,62,1)
# date_intrare_test_bilabial=date_intrare_test_bilabial.reshape(-1,62,62,1)

# date_intrare_antrenare_nasal=date_intrare_antrenare_nasal.reshape(-1,62,62,1)
# date_intrare_test_nasal=date_intrare_test_nasal.reshape(-1,62,62,1)

# date_intrare_antrenare_cv=date_intrare_antrenare_cv.reshape(-1,62,62,1)
# date_intrare_test_cv=date_intrare_test_cv.reshape(-1,62,62,1)

# date_intrare_antrenare_uw=date_intrare_antrenare_uw.reshape(-1,62,62,1)
# date_intrare_test_uw=date_intrare_test_uw.reshape(-1,62,62,1)

date_intrare_antrenare_iy=date_intrare_antrenare_iy.reshape(-1,62,62,1)
date_intrare_test_iy=date_intrare_test_iy.reshape(-1,62,62,1)


# x=date_intrare_antrenare_bilabial
# x=date_intrare_antrenare_nasal
# x=date_intrare_antrenare_cv
x=date_intrare_antrenare_iy
# x=date_intrare_antrenare_uw


# y=date_iesire_antrenare_aux_bilabial
y=date_iesire_antrenare_aux_iy
# y=date_iesire_antrenare_aux_uw
# y=date_iesire_antrenare_aux_cv
# y=date_iesire_antrenare_aux_nasal

# x_test=date_intrare_test_bilabial
x_test=date_intrare_test_iy
# x_test=date_intrare_test_uw
# x_test=date_intrare_test_nasal
# x_test=date_intrare_test_cv

# y_test=date_iesire_test_aux_bilabial
y_test=date_iesire_test_aux_iy
# y_test=date_iesire_test_aux_uw
# y_test=date_iesire_test_aux_cv
# y_test=date_iesire_test_aux_nasal

model.fit(  x
          , y
          ,validation_data = (x_test, y_test)
#           , validation_split=0.1
          , batch_size=64
          , epochs=50
          , shuffle=True
          , verbose=2)


# In[233]:



plt.imshow(date_intrare_antrenare[100])


# In[144]:


def crosscorr_2d(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """
    PRNU 2D cross-correlation
    :param k1: 2D matrix of size (h1,w1)
    :param k2: 2D matrix of size (h2,w2)
    :return: 2D matrix of size (max(h1,h2),max(w1,w2))
    """
    assert (k1.ndim == 2)
    assert (k2.ndim == 2)

    max_height = max(k1.shape[0], k2.shape[0])
    max_width = max(k1.shape[1], k2.shape[1])

    k1 -= k1.flatten().mean()
    k2 -= k2.flatten().mean()

    k1 /= np.linalg.norm(k1)
    k2 /= np.linalg.norm(k2)

    k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1])], mode='constant', constant_values=0)
    k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1])], mode='constant', constant_values=0)

    k1_fft = fft2(k1, )
    k2_fft = fft2(np.rot90(k2,2), )

    return np.roll(np.real(ifft2(k1_fft * k2_fft)).astype(np.float32),[-1,-1])


# In[5]:


test_loss, test_acc = model.evaluate(x_test, y_test)


# In[269]:


pred = model.predict(x)
print(pred.argmax(axis=1))


# In[212]:


pred[0]


# In[405]:


# saving the neural network

history = model.fit(x, y, batch_size=8, epochs=50, 
                    validation_data=(x_test, y_test))
# validation_data = (X_valid, y_valid)
model.save("Model_1_iy")


# In[198]:




# date_intrare_antrenare=[]
# date_intrare_test=[]
# date_iesire_antrenare=[]
# date_iesire_test=[]

# covariatie=np.load("eeg_filtrat_standardizat.npy")
# vector_bilabial=np.load("vector_bilabial.npy")

# shuffle= np.arange(covariatie.shape[0])
# np.random.shuffle(shuffle)
# covariatie= covariatie[shuffle]
# vector_bilabial= vector_bilabial[shuffle]

# date_intrare_antrenare, date_intrare_test, date_iesire_antrenare, date_iesire_test = train_test_split(covariatie, vector_bilabial, test_size=0.2,random_state=42,shuffle=True)

# # print(shuffle)
# covariatie.shape

# # print(date_intrare_antrenare)
# # print(date_iesire_antrenare)

