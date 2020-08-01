import pandas as pd
import emlearn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

csv_file='CSV/'
tflite_model_name = 'soil_model'  # Will be given .tflite suffix
c_model_name = 'soil_model'   # Will be given .h suffix

def hex_to_c_array(hex_data, var_name):

  c_str = ''

  # Create header guard
  c_str += '#ifndef ' + var_name.upper() + '_H\n'
  c_str += '#define ' + var_name.upper() + '_H\n\n'

  # Add array length at top of file
  c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

  # Declare C variable
  c_str += 'unsigned char ' + var_name + '[] = {'
  hex_array = []
  for i, val in enumerate(hex_data) :

    # Construct string from hex
    hex_str = format(val, '#04x')

    # Add formatting so each line stays within 80 characters
    if (i + 1) < len(hex_data):
      hex_str += ','
    if (i + 1) % 12 == 0:
      hex_str += '\n '
    hex_array.append(hex_str)

  # Add closing brace
  c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

  # Close out header guard
  c_str += '#endif //' + var_name.upper() + '_H'

  return c_str

dados = pd.read_csv(csv_file + 'data_300v1.csv', header=None)

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

#cmodel = emlearn.convert(classificador)
#cmodel.save(file='soilclassifier.h')

# Convert Keras model to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(classificador)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open(tflite_model_name + '.tflite', 'wb').write(tflite_model)


# Write TFLite model to a C source (or header) file
with open(c_model_name + '.h', 'w') as file:
  file.write(hex_to_c_array(tflite_model, c_model_name))

#classifier_json = classificador.to_json()
#with open('classificador_soil.json', 'w') as json_file:
#    json_file.write(classifier_json)
#classificador.save_weights('classificador_soil.h5')
