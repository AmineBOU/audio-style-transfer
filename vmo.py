
# coding: utf-8

# In[1]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[13]:

import numpy as np
import vmo.analysis as van
import vmo.generate as vge
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import librosa, vmo
import IPython.display
get_ipython().magic('matplotlib inline')


# In[67]:

# Setup
target_file = 'guitar.wav'
query_file = 'chopin.wav'

fft_size = 8192*4
hop_size = fft_size/2


# In[84]:

# Read target wave file 
y, sr = librosa.load(target_file, sr = 44100)
print(y)
C = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=fft_size, hop_length=int(hop_size))
#C = librosa.feature.chromagram(y=y, sr=sr, n_fft=fft_size, hop_length=hop_size, octwidth = None)
feature = np.log(C + np.finfo(float).eps)
feature = pre.normalize(feature)


# In[69]:

IPython.display.Audio(data=y, rate=sr)


# In[70]:

plt.figure(figsize = (12,2))
plt.imshow(feature, aspect = 'auto', origin = 'lower', interpolation = 'nearest', cmap ='Greys')
plt.title('Chromagram (target)', fontsize = 18)
plt.xlabel('Frame', fontsize = 14)
plt.ylabel('Chroma Bin', fontsize = 14)
plt.tight_layout()


# In[71]:

# Build target oracle
chroma_frames = feature.transpose()
r = (0.0, 1.01, 0.01) 
ideal_t = vmo.find_threshold(chroma_frames, r = r, flag = 'a', dim=chroma_frames.shape[1])
oracle_t = vmo.build_oracle(chroma_frames, flag = 'a', 
                            threshold = ideal_t[0][1], 
                            feature = 'chroma', dim=chroma_frames.shape[1])


# In[72]:

x = np.array([i[1] for i in ideal_t[1]])
y = [i[0] for i in ideal_t[1]]
fig = plt.figure(figsize = (12,4))
plt.plot(x, y, linewidth = 2)
plt.title('IR vs. Threshold Value(vmo)', fontsize = 16)
plt.grid(b = 'on')
plt.xlabel('Threshold')
plt.ylabel('IR')
plt.xlim(0,0.25)
plt.tight_layout()


# In[73]:

min_len = 1
pattern = van.find_repeated_patterns(oracle_t, lower = min_len)
pattern_mat = np.zeros((len(pattern), oracle_t.n_states-1))
for i,p in enumerate(pattern):
    length = p[1]
    for s in p[0]:
        pattern_mat[i][s-length:s-1] = 1

plt.figure(figsize = (12,2))
plt.imshow(pattern_mat, interpolation = 'nearest', aspect = 'auto', cmap = 'Greys')
plt.title('Patterns Found with VMO',fontsize=16)
# plt.yticks(np.arange(pattern_mat.shape[0]))
plt.ylabel('Pattern Index')
plt.xlabel('Frame Numbers',fontsize=16)
plt.tight_layout()


# In[74]:

# Read query wave file
y_q, sr = librosa.load(query_file, sr = 44100)
C_q = librosa.feature.chroma_stft(y=y_q, sr=sr, n_fft=fft_size, hop_length=int(hop_size))
feature_q = np.log(C_q+np.finfo(float).eps)
feature_q = pre.normalize(feature_q)


# In[75]:

IPython.display.Audio(data=y_q, rate=sr)


# In[76]:

plt.figure(figsize = (12,2))
plt.imshow(feature_q, aspect = 'auto', origin = 'lower', interpolation = 'nearest', cmap ='Greys')
plt.title('Chromagram (query)', fontsize = 18)
plt.xlabel('Frame', fontsize = 14)
plt.ylabel('Chroma Bin', fontsize = 14)
plt.tight_layout()


# In[77]:

# Query-matching and re-synthesis 
path, cost, i_hat = van.query(oracle_t, feature_q.T, trn_type = 1)
path = list(path)

x, _w, sr = vge.audio_synthesis(target_file, 'AllStar1.wav', path[i_hat], 
                                analysis_sr=sr, buffer_size=int(fft_size), hop=int(hop_size))


# In[78]:

IPython.display.Audio(data=x, rate=sr)


# In[79]:


print(x[:y_q.shape[0]].shape)

def getOneChannel(z):
    z = np.transpose(z)
    z = z[0]
    return z

x = getOneChannel(x)
print(y_q.shape)


# In[80]:

audio_mixed = np.vstack((x[:y_q.shape[0]]/float(np.max(np.abs(x))), y_q))
xnorm = x[:y_q.shape[0]]/float(np.max(np.abs(x)))
print(audio_mixed)
print(y_q.shape)
print(xnorm.shape)
#print(audio_mixed.shape)


# In[ ]:

IPython.display.Audio(data=audio_mixed, rate=sr)


# In[ ]:

print (path[i_hat])


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



