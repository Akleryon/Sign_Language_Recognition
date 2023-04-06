# Create CNN here
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras as kr
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

labels = ['01_palm/', '02_l/', '03_up/', '04_fist_moved/', '05_down/', '06_index/', '07_ok/', '08_palm_m/', '09_c/', '10_palm_u/', '11_heavy/', '12_hang/', '13_two/', '14_three/', '15_four/', '16_five/']
counter = 0
df = pd.read_csv('/Users/achilleraffin-marchetti/git/Data_Science_Projects/Sign_Language_Recognition/Code/Model/out.csv')

df2 = df.copy()
df2['Sign'] = df2['Sign'].map({'01_palm/':0, '02_l/':1, '03_up/':2, '04_fist_moved/':3, '05_down/':4, '06_index/':5, '07_ok/':6, '08_palm_m/':7, '09_c/':8, '10_palm_u/':9, '11_heavy/':10, '12_hang/':11, '13_two/':12, '14_three/':13, '15_four/':14, '16_five/':15})

y = df2.iloc[:,-1].values
X = df2.iloc[:, :-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=365)

train=tf.data.Dataset.from_tensor_slices((((X_train)),(y_train)))
valid=tf.data.Dataset.from_tensor_slices((((X_valid)),(y_valid)))
test=tf.data.Dataset.from_tensor_slices(((X_test)))

train_final=train.shuffle(X_train.shape[0]).batch(256).cache().prefetch(tf.data.AUTOTUNE)
valid_final=valid.batch(256).cache().prefetch(tf.data.AUTOTUNE)

model = kr.Sequential(
    [
        kr.layers.Input(63,name="Inputs"),
        kr.layers.Dense(63,activation="relu",name="Dense_layer1"),
        kr.layers.Dense(60,activation="relu",name="Dense_layer2"),
        kr.layers.Dense(50,activation="relu",name="Dense_layer3"),
        kr.layers.Dense(40,activation="relu",name="Dense_layer4"),
        kr.layers.Dense(30,activation="relu",name="Dense_layer5"),
        kr.layers.Dense(20,activation="relu",name="Dense_layer6"),
        kr.layers.Dense(16, activation='softmax', name='outputs')
    ]
)

model.compile(optimizer="Adam",loss=kr.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])
model.fit(train_final, validation_data=valid_final, epochs=200)

y_predict=model.predict(test.batch(1024))
predict=np.argmax(y_predict,axis=-1)

cm = confusion_matrix(y_test, predict)
print(cm)
print(accuracy_score(y_test, predict))

pickle.dump(model, open('model_file' + '.pkl', 'wb'))
