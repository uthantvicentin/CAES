#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 08:53:27 2020

@author: usuario
"""
import pandas as pd
import numpy as np

for i in range(7,10):
    ii = str(i)
    for j in range(7):
        j = j + 1
        caminho = ''
        aux = str(j)
        caminho = 'Documentos/PIBIC/Leonardo/dados/vermelho/vermelho_' + ii + '/vermelho_' + aux
        resul = pd.read_csv(caminho + '_.CSV', header=None, delimiter=',').loc[:,4]
        caminho += '_.csv'
        np.savetxt(fname=caminho, X=resul, fmt="%.5f", delimiter=",")
        print('OK')