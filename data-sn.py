import os
import glob
import random as rd 
from tqdm import tqdm
import torchaudio
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def writeDataDev(path, data_save):
    spk_id_south = os.listdir(path + "/south")
    spk_id_north = os.listdir(path + "/north")
    spk_id = spk_id_south + spk_id_north
    
    dev_list = []
    for id in spk_id:
        if id in spk_id_south:
            au_south = glob.glob(path + "south/" + id + "/*.wav")
            if len(au_south) > 3:
                full_wav1 = rd.sample(au_south, 3)
                dev_list.extend(full_wav1)
            else:
                dev_list.extend(au_south)
        else:
            au_north = glob.glob(path + "north/" + id + "/*.wav")
            if len(au_north) > 3:
                full_wav2 = rd.sample(au_north, 3)
                dev_list.extend(full_wav2)
            else:
                dev_list.extend(au_north)
        
    print(len(dev_list))
      
    with open(data_save + "data-st-dev.csv", "w") as wf:
        wf.writelines("\n".join(dev_list))
            
    return dev_list
            
def wrtieDataTrain(path, data_save, dev_list):
    
    spk_id_south = os.listdir(path + "/south")
    spk_id_north = os.listdir(path + "/north")
    spk_id = spk_id_south + spk_id_north
    
    train_list = []
    for id in spk_id:
        #import pdb
        #pdb.set_trace()
        if id in spk_id_south:
            au_south = glob.glob(path + "south/" + id + "/*.wav")
            if len(au_south) > 10 and au_south not in dev_list:
                full_wav1 = rd.sample(au_south, 10)
                train_list.extend(full_wav1)
            else:
                train_list.extend(au_south)
        else:
            au_north = glob.glob(path + "north/" + id + "/*.wav")
            if len(au_north) > 10 and au_north not in dev_list:
                full_wav2 = rd.sample(au_north, 10)
                train_list.extend(full_wav2)
            else:
                train_list.extend(au_north)
            
    
    with open(data_save + "data-st-train.csv", "w") as wf:
        wf.writelines("\n".join(train_list))
            
def modifiedDataTrain(data_save):
    wav = []
    start = []
    stop = []
    au = []
    dura = []
    spk_id = []
    ID = []
    data = []
    
    with open(data_save + "data-st-train.csv", "r") as rf:
        au = rf.read().splitlines()
    
    #print(au)
    #print(len(au))
    for i in tqdm(range(len(au))):
        
        signal, sample_rate = torchaudio.load(au[i])
        duration = float((signal.shape[1])/sample_rate)
        
        
        total_steps = (signal.shape[1]) // 48000
        
        if (duration - (total_steps*3.0)) > 1.0:
            total_steps += 1
            
        spk = os.path.basename(os.path.dirname(au[i]))
        au_id = os.path.splitext(os.path.basename(au[i]))[0]
        
        if duration < 3.0 and duration > 1.0:
                
            id_pos = f"{au_id}__00"
                
            ID.append(id_pos)
            start.append(str(0))
            stop.append(str(48000))
            wav.append(au[i])
            dura.append(str(duration))
            spk_id.append(spk)
                
        else:
            if (duration % 3.0) > 1.0 or duration % 3.0 == 0.0 :
                for step in range(total_steps):
                        
                    start1 = step * 48000
                    stop1 = (step + 1) * 48000
                        
                    id_pos = f"{au_id}__{step+1}"
                        
                    ID.append(id_pos)
                    start.append(str(start1))
                    stop.append(str(stop1))
                    wav.append(au[i])
                    dura.append(str(duration))
                    spk_id.append(spk)
                    
            
    for pos in tqdm(range(len(dura))):
        data.append(','.join([ID[pos], dura[pos], wav[pos], start[pos], stop[pos], spk_id[pos]]))
        
    unqdata = np.unique(data)
        
    # Write the modified data to a file
    with open(data_save + 'data-st-train.csv', 'w') as f:
        header = ['ID', 'duration', 'wav', 'start', 'stop', 'spk_id']
        file = csv.DictWriter(f, delimiter=',', fieldnames=header)
        file.writeheader()
        f.writelines('\n'.join(unqdata))

    '''
    # Load the modified data from the file
    df = pd.read_csv(data_save + 'data-st-train.csv')

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    # Save the train and test data to separate CSV files
    train_df.to_csv(data_save + 'train-south-north.csv', index=False)
    test_df.to_csv(data_save + 'dev-south-north.csv', index=False)
    '''
    
def modifiedDataDev(data_save):
    wav = []
    start = []
    stop = []
    au = []
    dura = []
    spk_id = []
    ID = []
    data = []
    
    with open(data_save + "data-st-dev.csv", "r") as rf:
        au = rf.read().splitlines()
    
    #print(au)
    #print(len(au))
    for i in tqdm(range(len(au))):
        
        signal, sample_rate = torchaudio.load(au[i])
        duration = float((signal.shape[1])/sample_rate)
        
        
        total_steps = (signal.shape[1]) // 48000
        
        if (duration - (total_steps*3.0)) > 1.0:
            total_steps += 1
            
        spk = os.path.basename(os.path.dirname(au[i]))
        au_id = os.path.splitext(os.path.basename(au[i]))[0]
        
        if duration < 3.0 and duration > 1.0:
                
            id_pos = f"{au_id}__00"
                
            ID.append(id_pos)
            start.append(str(0))
            stop.append(str(48000))
            wav.append(au[i])
            dura.append(str(duration))
            spk_id.append(spk)
                
        else:
            if (duration % 3.0) > 1.0 or duration % 3.0 == 0.0 :
                for step in range(total_steps):
                        
                    start1 = step * 48000
                    stop1 = (step + 1) * 48000
                        
                    id_pos = f"{au_id}__{step+1}"
                        
                    ID.append(id_pos)
                    start.append(str(start1))
                    stop.append(str(stop1))
                    wav.append(au[i])
                    dura.append(str(duration))
                    spk_id.append(spk)
                    
            
    for pos in tqdm(range(len(dura))):
        data.append(','.join([ID[pos], dura[pos], wav[pos], start[pos], stop[pos], spk_id[pos]]))
        
    unqdata = np.unique(data)
        
    # Write the modified data to a file
    with open(data_save + 'data-st-dev.csv', 'w') as f:
        header = ['ID', 'duration', 'wav', 'start', 'stop', 'spk_id']
        file = csv.DictWriter(f, delimiter=',', fieldnames=header)
        file.writeheader()
        f.writelines('\n'.join(unqdata))
        
        
def pairedData(data_save, pair_link):
    wav1 = []
    wav2 = []

    with open(data_save + "data-st-dev.csv", 'r') as f:
        lines = f.read().splitlines()

        for line in lines:
            wav1.append(line)

    wav2 = rd.sample(wav1, len(wav1))
    label = []
    pairs = []
    for id1 in tqdm(range(len(wav1))):
        spk1 = os.path.basename(os.path.dirname(wav1[id1]))
        for id2 in range(len(wav2)):
            au2 = wav2[id2]
            spk2 = os.path.basename(os.path.dirname(wav2[id2]))

            if spk1 == spk2:  # Compare speakers without using id2 index
                label.append(str(1))
                pairs.append(','.join([wav1[id1], au2, label[id1]]))
            else:
                label.append(str(0))
                pairs.append(','.join([wav1[id1], au2, label[id1]]))

            
        
    list_label = []
    pairs0 = []
    pairs1 = []
    wav_ele1 = []
    wav_ele2 = []
    with open(pair_link, 'w') as out1:
        for pair in tqdm(pairs):
            out1.write(pair+'\n')
    
    with open(pair_link, 'r') as out2:
        lines = out2.read().splitlines()
        for line in tqdm(lines):
            wav1, wav2, label = line.split(',')
            list_label.append(label)
            wav_ele1.append(wav1)
            wav_ele2.append(wav2)
            
        for id in tqdm(range(len(list_label))):
            if list_label[id] == str(0):
                pairs0.append(','.join([wav_ele1[id], wav_ele2[id], list_label[id]]))
            else:
                pairs1.append(','.join([wav_ele1[id], wav_ele2[id], list_label[id]]))
        
        print(len(pairs0))
        print(len(pairs1))
        
        new_pairs_1 = []
        n = 50000
        if len(pairs1) > 0:
            sample_size_1 = int(n * 0.3 + 1)
            if sample_size_1 > len(pairs1):
                sample_size_1 = len(pairs1)
            
        new_pairs_1 = rd.sample(pairs1, sample_size_1)
        
        new_pairs_0 = []
        if len(pairs0) > 0:
            sample_size_0 = int(n * 0.7)
            if sample_size_0 > len(pairs0):
                sample_size_0 = len(pairs0)
            new_pairs_0 = rd.sample(pairs0, sample_size_0)
            
        new_pairs = new_pairs_0 + new_pairs_1
        print(len(new_pairs))
    
    with open(pair_link, 'w') as out3:
        for pair in tqdm(new_pairs):
            out3.write(pair+'\n')
    
if __name__ == '__main__':
    path = "/data/processed/speech/spk_embedding/"
    data_save = "/project/AI-team/exp/experiments/phuonganh/data_audio/"
    pair_link = '/project/AI-team/exp/experiments/phuonganh/data_audio/dev-st-paired.csv'
    #dev_data = writeDataDev(path, data_save)
    #wrtieDataTrain(path, data_save, dev_data)
    #modifiedDataTrain(data_save)
    modifiedDataDev(data_save)
    #pairedData(data_save,pair_link)
