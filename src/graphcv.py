import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
import scikitplot as skplt
from sklearn.model_selection import train_test_split

#csv_file = '/content/drive/My Drive/UNIR/PIBIC/CAES/Amostras/Dados/Tres_Solos/'

csv_file='resultados/'

dados = pd.read_csv(csv_file + 'data_300.csv', header=None)

previsores = dados.iloc[:, 0:28].values
classe = dados.iloc[:, 28].values


le = LabelEncoder()
classe = le.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

previsorestrain, previsorestest, classetrain, classetest = train_test_split(previsores, classe, test_size = 0.2)

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def ANN():
    classificador = Sequential()
    classificador.add(Dense(units = 56, activation = 'relu', 
                            input_dim = 28))
    classificador.add(Dropout(0.2))
       
    for i in range(6):
      classificador.add(Dense(units = 56, activation = 'relu'))
      classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = 'rmsprop', 
                          loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificador

classifier = KerasClassifier(build_fn = ANN, epochs = 500, batch_size = 2)

resultados = cross_val_score(estimator = classifier, X = previsores,
                             y = classe, scoring = 'accuracy')

history = classifier.fit(previsores, classe, validation_split = 0.3, batch_size = 2)
predictions = cross_val_predict(classifier, previsores, classe)
skplt.metrics.plot_confusion_matrix(classe, predictions, normalize=True)
plt.show()

classifier_json = resultados.to_json()
with open('classificador_soil.json', 'w') as json_file:
    json_file.write(classifier_json)
resultados.save_weights('classificador_soil.h5')

#probas = classifier.predict_proba(previsorestest)
#skplt.metrics.plot_precision_recall_curve(classetest, probas)
#plt.show()
plot_history(history)

media = resultados.mean()
desvio = resultados.std()
print(media)
print(desvio)
