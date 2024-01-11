import os 
import numpy as np
from tqdm import tqdm 
import csv

def checkDataSNDev(data_path):
    au1 = [] 
    au2 = []
    label = []
    with open(data_path + "/dev-st-paired.csv", 'r') as f:
        lines = f.read().splitlines()
            
        for line in lines:
            ele1, ele2, lb = line.split(",")
            au1.append(ele1)
            au2.append(ele2)
            label.append(lb)
        
        count_num1 = 0
        count_num0 = 0
        spk_list1 = []
        spk_list2 = []
        for id in tqdm(range(len(label))):
            if label[id] == '0':
                count_num0 += 1
            else:
                count_num1 += 1
            
            spk1 = os.path.basename(os.path.dirname(au1[id]))
            spk2 = os.path.basename(os.path.dirname(au2[id]))
            spk_list1.append(spk1)
            spk_list2.append(spk2)
            total_spk = spk_list1 + spk_list2
        unique_spk = np.unique(total_spk)
        temp = f'st pairs: {len(label)}, st spks: {len(unique_spk)}, st0: {count_num0}, st1:{count_num1}'
        print(temp)
        
def checkDataZaloDev(data_path):
    au1 = [] 
    au2 = []
    label = []
    with open(data_path + "/dev-zalo-paired.csv", 'r') as f:
        lines = f.read().splitlines()
            
        for line in lines:
            ele1, ele2, lb = line.split(",")
            au1.append(ele1)
            au2.append(ele2)
            label.append(lb)
        
        count_num1 = 0
        count_num0 = 0
        spk_list1 = []
        spk_list2 = []
        for id in tqdm(range(len(label))):
            if label[id] == '0':
                count_num0 += 1
            else:
                count_num1 += 1
            
            spk1 = os.path.basename(os.path.dirname(au1[id]))
            spk2 = os.path.basename(os.path.dirname(au2[id]))
            spk_list1.append(spk1)
            spk_list2.append(spk2)
            total_spk = spk_list1 + spk_list2
        unique_spk = np.unique(total_spk)
        temp = f'zalo pairs: {len(label)}, zalo spks: {len(unique_spk)}, zalo0: {count_num0}, zalo1:{count_num1}'
        print(temp)
        
def checkDataSNTrain(data_path):
    spks = []
    with open(data_path + "data-st-train.csv", "r") as r:
        lines = r.read().splitlines()[1:]
        
        for line in tqdm(lines):
            _, _, _, _, _, spk_id = line.split(",")
            spks.append(spk_id)
        
    total_spk = np.unique(spks)
    temp = f"st wav:{len(lines)}, st spks: {len(total_spk)}"
    print(temp)
    
def checkDataZaloTrain(data_path):
    spks = []
    with open(data_path + "data-zalo-train.csv", "r") as r:
        lines = r.read().splitlines()[1:]
        
        
        for line in tqdm(lines):
            _, _, _, _, _, spk_id = line.split(",")
            #print(spk_id)
            spks.append(spk_id)
   
    total_spk = np.unique(spks)
    temp = f"zalo wav:{len(lines)}, zalo spks: {len(total_spk)}"
    print(temp)
    #print(lines)
    
        
if __name__ == '__main__':
    data_path = "/project/AI-team/exp/experiments/phuonganh/data_audio/"
    checkDataSNDev(data_path)
    checkDataSNTrain(data_path)
    checkDataZaloDev(data_path)
    checkDataZaloTrain(data_path)