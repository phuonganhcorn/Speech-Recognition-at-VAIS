import os
import numpy as np
from tqdm import tqdm
import glob
import random as rd
import torchaudio
import pandas as pd
import torch
import csv
from sklearn.model_selection import train_test_split
from speechbrain.pretrained import EncoderClassifier


if __name__ == "__main__":
    dura = []
    speaker_id = []
    start = []
    stop = []
    data = []
    wav_path = []
    id = []
    
    ''' list id in dataset folder'''
    path = '/data/processed/speech/zalo_challenge/dataset'
    id_path = glob.glob("/data/processed/speech/zalo_challenge/dataset/*/*.wav")
    
    count_speaker = os.listdir(path)
    print("num of speaker =", len(count_speaker))
    
    ''' get signal, samplerate '''
    for i in tqdm(range(len(id_path))):
        
        wav_file_name = os.path.splitext(os.path.basename(id_path[i]))[0]
        
        signal, sample_rate = torchaudio.load(id_path[i])
        duration = float((signal.shape[1])/sample_rate)
    
        total_steps = (signal.shape[1]) // 48000
        
        parent_dir = os.path.basename(os.path.dirname(id_path[i]))
            
        if duration >= 3.0:
            for step in range(total_steps):
                
                start1 = step * 48000
                stop1 = (step + 1) * 48000
                
                
                id_pos = f"{parent_dir}__{wav_file_name}__{step+1}"
                
                id.append(id_pos)
                start.append(str(start1))
                stop.append(str(stop1))
                wav_path.append(id_path[i])
                dura.append(str(duration))
                speaker_id.append(parent_dir)
                
                
        if duration < 3.0 and duration > 1.0:
            
            id_pos = f"{parent_dir}__{wav_file_name}__00"
            
            id.append(id_pos)
            start.append(str(0))
            stop.append(str(48000))
            wav_path.append(id_path[i])
            dura.append(str(duration))
            speaker_id.append(parent_dir)
            
            
        else:
            continue
            
    for pos in tqdm(range(len(dura))):
        data.append(','.join([id[pos], dura[pos], wav_path[pos], start[pos], stop[pos], speaker_id[pos]]))
        
    #print(speaker_id)
    
    ''' write train file '''
    header = ['ID', 'duration', 'wav', 'start', 'stop', 'spk_id']
        
    with open ('train-zalo.csv', 'w') as f:
        file = csv.DictWriter(f, delimiter = ',', fieldnames = header)
        file.writeheader()
        f.writelines('\n'.join(data))
    
    df = pd.read_csv('train-zalo.csv')
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    ''' save the train and test data to separate CSV files '''
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('dev.csv', index=False)
    
       
    
    
    #print(len(sr))
    
    
                                
                        