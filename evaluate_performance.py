import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from build_data import test 

model = tf.keras.models.load_model('model.h5')

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}")