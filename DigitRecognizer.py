'''
link: https://www.kaggle.com/c/digit-recognizer
'''

#### Digit Recognition (Kaggle) ######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle



#Read data
data = pd.read_csv('data\train.csv')
test = pd.read_csv('data\test.csv')
data = shuffle(data, random_state=0)

#preprocessing
train  = data.drop('label', axis=1)
target = data['label']

#target one-hot encoding
one_hot_target = pd.get_dummies(target)

#Training
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropie = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropie)


#Lunch the model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batchx = train[:100]
    batchy = one_hot_target[:100]
    sess.run(train_step, feed_dict={X: batchx, y_: batchy})

#Predictions
prediction = tf.argmax(y, 1)
predictions = sess.run(prediction, feed_dict={X: test})


#Edit submission file
li = [i+1 for i in range(len(test))]

submission = pd.DataFrame({
        "ImageId": li,
        "Label": predictions
     })

# Any files you save will be available in the output tab below
submission.to_csv('submission.csv', index=False)




