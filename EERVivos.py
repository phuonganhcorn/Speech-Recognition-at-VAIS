import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import os
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import random as rd 
import csv

def modifyVivos(raw_direct_path):
    vivos_subdir = []
    vivos_file =[]
    vivos = []
    au_data = []
    label_list = []

    rootdir = '/home/phuonganh/speechbrain/vivos/test/waves'
    for it in os.listdir(rootdir):
        d = os.path.join(rootdir, it)
        if os.path.isdir(d):
            vivos_subdir.append(d)  # vivos_subdir = folder of speakers
    #print(vivos_subdir)


    for dir in vivos_subdir:
        vivos_audio_dir= os.path.join(rootdir, dir)
        vivos.append(vivos_audio_dir)   # vivos = list of full audio's linkpath

    for audir in vivos:
        files = os.listdir(audir)
        files = [f for f in files if os.path.isfile(audir + '/' + f)]

        #print(*files, sep = '\n')

        for file in files:
            with open("modified_vivos_data.csv", "w") as w:
                w.writelines("\n".join(files))

        with open("modified_vivos_data.csv", "r") as f:
            lines = f.read().splitlines()

            for line in lines:
                vivos_audio = os.path.join(audir, line)
                vivos_file.append(vivos_audio)
                
                                #print(vivos_file)

            # shuffle audios in vivos_file for random location
            element1 = rd.sample(vivos_file, len(vivos_file))
            #print(element1)

            # shuffle second time for second element
            element2 = rd.sample(vivos_file, len(vivos_file))


        '''
        Label data
        If speaker of audio in elemetn1 = audio in element2
            label = 1
        else:
            label in range(len(element1)):
            au1 = element1[i]
            au2 = element2[i]

            folder1 = os.path.dirname(au1)
            folder2 = os.path.dirname(au2)

            if folder1 == folder2:
                label_list.append(1)
            else:
                label_list.append(0)

            au_data.append(','.join([element1[i], element2[i], str(label_list[i])]))

        #print(au_data)  

    with open("modified_vivos_data.csv", "w") as w:
        w.writelines('\n'.join(au_data))
        '''
                                                             





def readAudioData(audio_data_path):
    # Initialize arr 
    element1 = []
    element2 = []
    label_list = []
    
    # read audio data file
    with open(audio_data_path, 'r') as f:
        
        lines = f.read().splitlines()
        
        for line in lines:
            audio1, audio2, label = line.split(",")
            element1.append(audio1) # list of audio1
            element2.append(audio2) # list of audio2
            label_list.append(int(label))
            
    return element1, element2, label_list



'''
    In calEmbeddings def
    
        1. Cal embeddings of each audio in arr unique_audio
        2. Put into an embedds[] arr
        3. Zip two audios to make a pair for comparing
        4. Visualize:
        
            [element1 [audio1, audio2, audio3,..., ...]
             element2 [audio1, audio2, audio3,..., ...]] 
            
        => compare index of ele in unique_audio to get embedd of that audio           
'''
def calEmbeddings(element1, element2):
    
    # initialize list storage val of embeddings
    val_embedds = []
    
    # load pre-trained model
    # verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="test_huggingface/vivos")
    classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts = {"device":"cuda:0"})

            
            
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
        signal1, fs1 = torchaudio.load(unique_audio[i])
        embeddings = classifier.encode_batch(signal1).squeeze().detach().cpu().numpy()
            
        # put embedds into arr
        val_embedds.append(embeddings)
            
        # signal2, fs2 = torchaudio.load(element2[i])
        # embeddings2 = classifier.encode_batch(signal2).squeeze().detach().cpu().numpy()
        # data2.append(embeddings2)
            
            
    # zip arr embedds and arr unique_audio
    # au_em = (("au1", "em1"), ("au2", "em2"), ...)
    pair_zip = zip(au_arr1, au_arr2)
        

    return pair_zip, val_embedds, unique_audio


'''     
    In calCosineDiff def
        1. Iterater each pair of element in zip array: contains index of audio                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        2. Iterater each element in unique_audio: contains index of embedds
        3. If unique[id] = arr[id] => get embedds[id]
        4. Cal cosine diff of 2 embedds 
'''
def cosineDiff(pair_zip, val_embedds, unique_audio):
    # Initialize embedd_score arr
    embedds_score = []

    # Get index of audio in unique_audio arr
    for id, (name_au1, name_au2) in tqdm(enumerate(pair_zip)):
        id1 = np.where(unique_audio == name_au1)[0][0]
        id2 = np.where(unique_audio == name_au2)[0][0]
        
        # get embedds of audio id1, id2
        val1 = val_embedds[id1]
        val2 = val_embedds[id2]
        
        
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
    print(eer)
    print(thresh)
    
            

if __name__ == "__main__":
    raw_vivos_path = '/home/phuonganh/speechbrain/vivos/test/waves/'
    audio_data_path = '/home/phuonganh/speechbrain/modified_vivos_data.csv'
    element1, element2, label_list = readAudioData(audio_data_path=audio_data_path)
    pair_zip, val_embedds, unique_audio = calEmbeddings(element1, element2)
    embedds_score = cosineDiff(pair_zip, val_embedds, unique_audio)
    tuningEER(y_true = label_list ,y_score = embedds_score)


vi 
