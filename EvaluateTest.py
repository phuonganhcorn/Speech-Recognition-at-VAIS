import torchaudio
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import os
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import io
import sys
import numpy as np
import torch.onnx
import torchaudio
from torch import  nn
from lobes.models.ECAPA_TDNN_192 import ECAPA_TDNN
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
import glob
import random as rd

class COMPUTE_EMB(nn.Module):
    def __init__(self, model_path):
        super(COMPUTE_EMB, self).__init__()
        self.model_path = model_path
        self.model = ECAPA_TDNN(input_size=80,
                                channels=[1024, 1024, 1024, 1024, 3072],
                                kernel_sizes=[5, 3, 3, 3, 1],
                                dilations=[1, 2, 3, 4, 1],
                                attention_channels=128,
                                lin_neurons=192)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.model = self.model.cuda()
        self.feature_maker = Fbank(n_mels=80)
        self.norm = InputNormalization(norm_type='sentence', std_norm=False)
    def forward(self, signal):
        signal = signal.cuda()
        sig_lens = torch.ones(signal.shape[0])
        feats = self.feature_maker(signal)
        feats = self.norm(feats, sig_lens)
        embeddings = self.model(feats)
        return embeddings


pair_link = '/project/AI-team/exp/experiments/phuonganh/data_audio/dev-st-paired.csv'
'''
    In calEmbeddings def
    
        1. Cal embeddings of each audio in arr unique_audio
        2. Put into an embedds[] arr
        3. Zip embedds[] and unique_audio[] 
        4. Visualize:
        
            [unique_audio[audio1, audio2, audio3,..., ...]
            embedds     [em_au1, em_au2, em_au3,..., ...]] 
            
        => compare index of ele in unique_audio to get embedd of that audio           
'''
        

def calEmbeddings(pair_link):
    # Initialize arr 
    element1 = []
    element2 = []
    label_list = []
    val_embedds = []
    
    # read audio data file
    with open (pair_link, 'r') as f:
        
        lines = f.read().splitlines()
        
        for line in tqdm(lines):
            audio1, audio2, label = line.split(",")
            element1.append(audio1) # list of audio1
            element2.append(audio2) # list of audio2
            label_list.append(int(label))
            
            
        # convert list audio into array
        au_arr1 = np.array(element1)
        au_arr2 = np.array(element2)
        # combine 2 list into 1 total list
        total_list = element1 + element2
        total_arr = np.array(total_list)
        # print(total_arr)
            
        # get unique audio 
        unique_audio = np.unique(total_arr) # arr of unique audio
        
        
        # cal embedds for each audio in unique_audio_arr
        for i in tqdm(range(len(unique_audio))):
            signal, fs = torchaudio.load(unique_audio[i])
            
            embeddings = compute_emb(signal).squeeze().detach().cpu().numpy()
            
            if embeddings.shape[0] == 2:
                #import pdb
                #pdb.set_trace()
                embeddings = embeddings[0,:]
            # put embedds into arr
            val_embedds.append(embeddings)
            
            # signal2, fs2 = torchaudio.load(element2[i])
            # embeddings2 = classifier.encode_batch(signal2).squeeze().detach().cpu().numpy()
            # data2.append(embeddings2)
            
            
        # zip arr embedds and arr unique_audio
        # au_em = (("au1", "em1"), ("au2", "em2"), ...)
        au_em = zip(au_arr1, au_arr2)
        

    return label_list, au_em, val_embedds, unique_audio


'''     
    In calCosineDiff def
        1. Iterater each element in arr1: contains audio1
        2. Iterater each element in arr2: contains audio2
        3. If unique[id] = arr[id] => get embedds[id]
        4. Cal cosine diff of 2 embedds 
'''
def cosineDiff(au_em, val_embedds, unique_audio):
    # Initialize embedd_score arr
    embedds_score = []



    # Get index of audio in unique_audio arr
    for id, (name_au1, name_au2) in enumerate(au_em):
        id1 = np.where(unique_audio == name_au1)[0][0]
        id2 = np.where(unique_audio == name_au2)[0][0]
        
        
        
        # get embedds of audio id1, id2
        val1 = val_embedds[id1]
        val2 = val_embedds[id2]
        
        # choose chanel
        #val1 = val1.[]
        
        # cal 2 embedds value
        diff = np.dot(val1, val2)/((norm(val1)*norm(val2)))
        embedds_score.append(diff)
    return embedds_score



# y_score = embedding
# y_true = label

def tuningEER(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh
    
            

if __name__ == "__main__":
    global compute_emb
    model_path = "/project/AI-team/exp/experiments/phuonganh/output/train/2001/save/CKPT+2023-09-26+15-22-08+00/embedding_model.ckpt"
    compute_emb = COMPUTE_EMB(model_path)
    label_list, au_em, val_embedds, unique_audio = calEmbeddings(pair_link)
    embedds_score = cosineDiff(au_em, val_embedds, unique_audio)
    eer, thresh = tuningEER(y_true = label_list ,y_score = embedds_score)
    print("eer: ", eer)
    print("thresh", thresh)
    print(model_path)
    
    
   


