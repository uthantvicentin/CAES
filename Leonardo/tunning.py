#@title Tunning 3 { run: "auto", vertical-output: true }
import pandas as pd
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#csv_file = '/content/drive/My Drive/UNIR/PIBIC/CAES/Amostras/Dados/Tres_Solos/'

csv_file='resultados/'

dados = pd.read_csv(csv_file + 'data_300.csv', header=None)

previsores = dados.iloc[:, 0:28].values
classe = dados.iloc[:, 28].values


le = LabelEncoder()
classe = le.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, input_dim = 28))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, loss = loos,
                          metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede, verbose = 1)

parametros = {
              'batch_size': [2],
              'epochs': [500],
              'optimizer': ['rmsprop'],
              'loos': ['sparse_categorical_crossentropy'],
              'kernel_initializer': ['random_uniform'],
              'activation': ['relu'],
              'neurons': [56,58]
}
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)



grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print(melhores_parametros)
print(melhor_precisao)
