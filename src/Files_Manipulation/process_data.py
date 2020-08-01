import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis

def repeat(list_):
  N = len(list_)
  temp = []
  for k in range(N):
      temp += [[list_[k]]]
  return temp

def process(args_):
    split = 250
    times = int(2500/split)
    path_ = args_[0]
    files = os.listdir(path_)
    files.sort()
    global case
    case += 1
    print("Case:", case)


    
    dataset = pd.DataFrame({'10Hz_Kurtosis': [], '100Hz_Kurtosis': [], 
                            '1.000Hz_Kurtosis': [], '10.000Hz_Kurtosis': [],
                            '100.000Hz_Kurtosis': [],'1.000.000Hz_Kurtosis':[],
                            '10.000.000Hz_Kurtosis': [],
                               
                            '10Hz_Mean': [], '100Hz_Mean': [], 
                            '1.000Hz_Mean': [], '10.000Hz_Mean': [],
                            '100.000Hz_Mean': [], '1.000.000Hz_Mean':[],
                            '10.000.000Hz_Mean': [],
                               
                            '10Hz_Std': [], '100Hz_Std': [], 
                            '1.000Hz_Std': [], '10.000Hz_Std': [],
                            '100.000Hz_Std': [], '1.000.000Hz_Std':[],
                            '10.000.000Hz_Std': [],
                               
                            '10Hz_Var': [], '100Hz_Var': [], 
                            '1.000Hz_Var': [], '10.000Hz_Var': [],
                            '100.000Hz_Var': [], '1.000.000Hz_Var':[],
                            '10.000.000Hz_Var': [],
                            
                            'Classe' : []})
    
         
    for filename in files:
        
    
      fixcsv = pd.read_csv(path_ + filename, header=None, delimiter=',').loc[:,4]
      resul = abs(np.fft.fft(fixcsv))
      
      pos = ''
    
      if filename[len(filename)-8] == '0':
          pos = '10'
      elif filename[len(filename)-8] == '1':
          pos = '100'
      elif filename[len(filename)-8] == '2':
          pos = '1.000'
      elif filename[len(filename)-8] == '3':
          pos = '10.000'
      elif filename[len(filename)-8] == '4':
          pos = '100.000'
      elif filename[len(filename)-8] == '5':
          pos = '1.000.000'
      else:
          pos = '10.000.000'
     
      temp = []
      aux = split
      for i in range(times):
          temp.append(float(kurtosis(resul[aux-split:aux-1])))
          aux += split
      dataset[pos + 'Hz_Kurtosis'] = temp
      
      temp = []
      aux = split
      for i in range(times):
          temp.append(float(np.mean(resul[aux-split:aux-1])))
          aux += split
      dataset[pos + 'Hz_Mean'] = temp
      
      temp = []
      aux = split
      for i in range(times):
          temp.append(float(np.std(resul[aux-split:aux-1], ddof = 1)))
          aux += split
      dataset[pos + 'Hz_Std'] = temp
      
      temp = []
      aux = split
      for i in range(times):
          temp.append(float(np.var(resul[aux-split:aux-1])))
          aux += split
      dataset[pos + 'Hz_Var'] = temp
    
   
    tt = pd.DataFrame({'10Hz_Kurtosis': [], '100Hz_Kurtosis': [], 
                            '1.000Hz_Kurtosis': [], '10.000Hz_Kurtosis': [],
                            '100.000Hz_Kurtosis': [], '1.000.000Hz_Kurtosis':[],
                            '10.000.000Hz_Kurtosis': [],
                               
                            '10Hz_Mean': [], '100Hz_Mean': [], 
                            '1.000Hz_Mean': [], '10.000Hz_Mean': [],
                            '100.000Hz_Mean': [], '1.000.000Hz_Mean':[],
                            '10.000.000Hz_Mean': [],
                               
                            '10Hz_Std': [], '100Hz_Std': [], 
                            '1.000Hz_Std': [], '10.000Hz_Std': [],
                            '100.000Hz_Std': [], '1.000.000Hz_Std':[],
                            '10.000.000Hz_Std': [],
                               
                            '10Hz_Var': [], '100Hz_Var': [], 
                            '1.000Hz_Var': [], '10.000Hz_Var': [],
                            '100.000Hz_Var': [], '1.000.000Hz_Var':[],
                            '10.000.000Hz_Var': [],
                            
                            'Classe' : []})
    
    tt = np.concatenate([dataset,tt])
    return tt

if __name__ == "__main__":
    directories = [dir_ for dir_ in os.listdir("../Data") if os.path.isdir("../")]
    case = 0
    results = []
    
    for dir_ in directories:
        path_relative = "../Data/" + dir_ + "/"
        sub_dir = [path_relative + x + "/" for x in os.listdir(path_relative)]
        sub_dir.sort()
        result =  map(process, repeat(sub_dir))
        results += result
    data = pd.DataFrame({'10Hz_Kurtosis': [], '100Hz_Kurtosis': [], 
                        '1.000Hz_Kurtosis': [], '10.000Hz_Kurtosis': [],
                        '100.000Hz_Kurtosis': [], '1.000.000Hz_Kurtosis':[],
                        '10.000.000Hz_Kurtosis': [],
                           
                        '10Hz_Mean': [], '100Hz_Mean': [], 
                        '1.000Hz_Mean': [], '10.000Hz_Mean': [],
                        '100.000Hz_Mean': [], '1.000.000Hz_Mean':[],
                        '10.000.000Hz_Mean': [],
                           
                        '10Hz_Std': [], '100Hz_Std': [], 
                        '1.000Hz_Std': [], '10.000Hz_Std': [],
                        '100.000Hz_Std': [], '1.000.000Hz_Std':[],
                        '10.000.000Hz_Std': [],
                           
                        '10Hz_Var': [], '100Hz_Var': [], 
                        '1.000Hz_Var': [], '10.000Hz_Var': [],
                        '100.000Hz_Var': [], '1.000.000Hz_Var':[],
                        '10.000.000Hz_Var': [],
                        
                        'Classe' : []})
    
    
    for i in range(30):
        data = np.concatenate([data,results[i]])
    for i in range(300):
        data[i][28] = 1
            
    if results != []:
      scaler = MinMaxScaler()
      data = scaler.fit_transform(data)
     
      for i in range(300):
        if i <= 99:
            data[i][28] = 0
        elif i <= 199:
            data[i][28] = 1
        else:
            data[i][28] = -1
    
    np.random.shuffle(data)
    np.savetxt(fname="../CSV/data_300.csv", X=data, fmt="%.15f", delimiter=",")
  
