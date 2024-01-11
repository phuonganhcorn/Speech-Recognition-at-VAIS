import os 
import random as rd
from tqdm import tqdm
import pdb 
import csv

def writeTest(audio_data_path, pair_link):
    wav = []
    spk_id = []
    
    with open(audio_data_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Bỏ qua header

        for row in csv_reader:
            id, _, wavlink, _, _, spk_id_val = row
            wav.append(wavlink)
            spk_id.append(spk_id_val)

    ele1 = [','.join([wavlink, spk_id_val]) for wavlink, spk_id_val in zip(wav, spk_id)]

    ele2 = rd.sample(ele1, len(ele1))
    
    with open('/project/AI-team/exp/experiments/phuonganh/data_audio/ele1.csv', 'w') as w1:
        for val in tqdm(ele1):
            w1.write(val + '\n')
    
    with open('/project/AI-team/exp/experiments/phuonganh/data_audio/ele2.csv', 'w') as w2:
        for val in tqdm(ele2):
            w2.write(val + '\n')
        
    wav_ele1 = []
    spk_ele1 = []
    with open('/project/AI-team/exp/experiments/phuonganh/data_audio/ele1.csv', 'r') as r1:
        lines = r1.read().splitlines()
        
        for line in lines:
            wav1, spk1 = line.split(',')
            wav_ele1.append(wav1)
            spk_ele1.append(spk1)
   
    wav_ele2 = []
    spk_ele2 = []
    with open('/project/AI-team/exp/experiments/phuonganh/data_audio/ele2.csv', 'r') as r2:
    
        lines = r2.read().splitlines()
        
        for line in lines:
            wav2, spk2 = line.split(',')
            wav_ele2.append(wav2)
            spk_ele2.append(spk2)
    

    label = []
    pairs = []      
    for id1 in tqdm(range(len(wav_ele1))):
        au1 = wav_ele1[id1]
        for id2 in range(len(wav_ele2)):
            au2 = wav_ele2[id2]
            
            if spk_ele1[id2] == spk_ele2[id2]:
                label.append(str(1))
                pairs.append(','.join([wav_ele1[id2], au2, label[id2]]))
            else:
                label.append(str(0))
                pairs.append(','.join([wav_ele1[id2], au2, label[id2]]))
            
        
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
            
            
def comparePairs():
    train_data = []
    test_data = []
    count_train_row = 0 
    count_test_row = 0
    count = 0
    with open('/project/AI-team/exp/experiments/phuonganh/data_audio/train.csv', 'r') as tf:
        csv_reader = csv.reader(tf)
        next(csv_reader)
        
        for row in csv_reader:
            train_data.append(row)
            count_train_row += 1
        
        
    with open('/project/AI-team/exp/experiments/phuonganh/data_audio/dev.csv', 'r') as df:
        csv_reader = csv.reader(df)
        next(csv_reader) 
        
        for row in csv_reader:
            test_data.append(row)
            count_test_row += 1
        
    for data in test_data:
        if data in train_data:
            count += 1
    
    print(count_train_row, count_test_row, count)
    
            
if __name__ == "__main__":
    
    audio_data_path = '/project/AI-team/exp/experiments/phuonganh/data_audio/dev.csv' # zalo challenge audio data
    pair_link = '/project/AI-team/exp/experiments/phuonganh/data_audio/dev_paired.csv'
    
    #dev_pairs = writeTest(audio_data_path, pair_link)
    comparePairs()
    
            
    
            