import pandas as pd
import emlearn
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

csv_file='resultados/'

dados = pd.read_csv(csv_file + 'data_300.csv', header=None)

previsores = dados.iloc[:, 0:28].values
classe = dados.iloc[:, 28].values


le = LabelEncoder()
classe = le.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()
classificador.add(Dense(units = 56, activation = 'relu', 
                            input_dim = 28))
classificador.add(Dropout(0.2))
       
for i in range(6):
    classificador.add(Dense(units = 56, activation = 'relu'))
    classificador.add(Dropout(0.2))

classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

classificador.fit(previsores, classe_dummy, epochs = 20, batch_size = 2)

cmodel = emlearn.convert(classificador)
cmodel.save(file='sonar.h')

classifier_json = classificador.to_json()
with open('classificador_soil.json', 'w') as json_file:
    json_file.write(classifier_json)
classificador.save_weights('classificador_soil.h5')