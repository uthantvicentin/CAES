#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from oct2py import octave
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def putinorder(files, order):
    temp = [""] * len(order)
    for i in range(len(order)):
        for file in files:
            if order[i] in file:
                temp[i] = file
                break
    return temp
    

def repeat(list_, arg_a, arg_b):
    N = len(list_)
    temp = []
    for k in range(N):
        temp += [[list_[k], arg_a, arg_b]]
    return temp


def process(args_):
    temp = []
    path_ = args_[0]
    inicio_ = args_[1]
    fim_ = args_[2]
    files = putinorder(os.listdir(path_), ["1", "2", "3", "4", "5", "6", "7"]) #7frequencias
    global case
    
    case += 1
    print("Case:", case)
    for filename in files:
        # Read file
        ca = pd.read_csv(path_ + filename, header=None, delimiter=',').iloc[:].values.transpose()[0]
        ca = ca[int(len(ca)*inicio_/40):int(len(ca)*fim_/40):4]
        ca = octave.frft2(ca, 1)
        ca = np.abs(ca)
        temp.append(np.std(ca, ddof = 1))
	
   
    if "amarelo" in filename:
        temp.append(2)
   
    else:
        temp.append(0)
    return temp


if __name__ == "__main__":
    directories = [dir_ for dir_ in os.listdir("dados") if os.path.isdir("dados")]
    case = 0
    
    
    intervalos =[[1, 20], [20, 40]]
    results = []
       
    for dir_ in directories:
        for intervalo in intervalos:
            path_relative = "dados/" + dir_ + "/"
            sub_dir = [path_relative + x + "/" for x in os.listdir(path_relative)]
            
            result = map(process, repeat(sub_dir, intervalo[0], intervalo[1]))
            results += result
            
    
    if results != []:
        
        train_data = results;
    
        # Normaliza entre 0 e 1
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
    
        # escreve os cvs
        np.savetxt(fname="resultados/train_data.csv", X=train_data, fmt="%.15f", delimiter=",")

