import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential
from concordance import concordance_index
from survival_function_est import get_survival_plot


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')

file_handler = logging.FileHandler('training_log.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)




class DeepMultiTasks:

    def __init__(self, df, label, event, elements=[32, 64], activation = ['relu', 'tanh'], X_col=None):
        self.df = df.sample(frac=1.0)
        y_col = [label, event]
        self.y_col = y_col
        # self.Y = df.get(y_col)
        self.label = np.array(self.df[label])
        self.event = np.array(self.df[event])
        # sort_idx = np.argsort(label)[::-1]
        if X_col != None:
            self.X = np.array(self.df[X_col]).astype("float64")
        else:
            self.X = np.array(self.df.drop(y_col, axis=1).astype("float64"))

        global MAX_MAT_COL
        MAX_MAT_COL = int(np.max(self.label) + 1)
        self.MAX_MAT_COL = MAX_MAT_COL
        self.triangle = self.get_triangle()
        self.model = self.nn_struct(elements, activation)


    def nn_struct(self, elements, activation):
        inputs = tf.keras.layers.Input([len(self.X[0]), ], name='input_X')
        inputs_label = tf.keras.layers.Input([1, ], name="input_label")
        inputs_nllmat = tf.keras.layers.Input([self.MAX_MAT_COL+1, ], name="input_nllmat")
        # inputs_rankmat = tf.keras.layers.Input([self.MAX_MAT_COL+1, ], name="input_rankmat")
        inputs_event = tf.keras.layers.Input([1, ], name="input_event")
        #inputs_triangle = tf.keras.layers.Input([self.MAX_MAT_COL+1, ], name="input_triangle")
        # count = 0
        # inputs_X = tf.keras.layers.BatchNormalization()(inputs)

        inputs_X = tf.keras.layers.Dense(elements[0], activation='relu')(inputs)
        for (i, j) in zip(elements[1:], activation[1:]):
            #if count % 2 == 0:
            inputs_X = tf.keras.layers.BatchNormalization()(inputs_X)
            inputs_X = tf.keras.layers.Dense(i, activation=j)(inputs_X)
            # else:
            #     inputs = tf.keras.layers.BatchNormalization()(inputs)
            #     inputs = tf.keras.layers.Dense(i, activation='relu')(inputs)
        # count = count + 1


        output = tf.keras.layers.BatchNormalization()(inputs_X)
        output = tf.keras.layers.Dense(self.MAX_MAT_COL, activation='linear')(output)

        outputs = Transform()([output, self.triangle])

        # tf.print(inputs_label.shape)
        # tf.print(inputs_nllmat.shape)
        # tf.print(inputs_event.shape)
        # tf.print(outputs.shape)

        my_loss = Total_Loss()([inputs_label, inputs_nllmat, inputs_event, outputs])

        model = tf.keras.models.Model(inputs=[inputs, inputs_label, inputs_nllmat, inputs_event],
                                      outputs=[outputs, my_loss])

        return model

    def train(self,
              epochs,
              batch_size = 256,
              initial_learning_rate=0.01,
              decay_steps=10000,
              decay_rate=0.9,
              summary=0
              ):
        self.batch_size = batch_size
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                               decay=0.0)
            # , loss=self.negative_log_likelihood(self.event)
            # , metrics=[concordance_metric(event = self.event)]
                          )


        data = self.get_traindata()
        self.model.fit(data,
                       epochs=epochs
                       # batch_size=1024
                       )

        if summary:
            self.model.summary()

    def get_traindata(self):
        def map_fn(X, label, nll_mat, event):#, triangle):
            inputs = {"input_X": X,
                      "input_label": label,
                      "input_nllmat": nll_mat,
                      "input_event": event,
                      # "input_triangle": triangle
                      }

            targets = {}
            return inputs, targets

        #     events = tf.convert_to_tensor(event, dtype=tf.float64)
        mat_tensor = self.get_matrix()
        # triangle = self.get_triangle()
        dataset = tf.data.Dataset.from_tensor_slices((self.X,
                                                      self.label,
                                                      mat_tensor,
                                                      self.event,
                                                      # triangle
                                                      )).map(map_fn).batch(self.batch_size)
        return dataset

    def get_triangle(self):
        triangle = np.tri(self.MAX_MAT_COL, self.MAX_MAT_COL+1, dtype=np.float32)
        return tf.convert_to_tensor(triangle)

    def get_matrix(self, label = None):
        if type(label) == type(None):
            label = self.label
        rows = len(label)
        mat = np.zeros([rows, MAX_MAT_COL+1])
        for i in range(rows):
            if self.event[i] != 0:
                mat[i, int(self.label[i])] = 1
            else:
                mat[i, int(self.label[i]) + 1:] = 1
        mat_tensor = (tf.convert_to_tensor(mat))

        ##

        return mat_tensor


    def predict_score(self, X):
        # print(X.shape)
        # print(self.X.shape)
        # print(self.label.shape)
        label = np.zeros(X.shape[0],)
        event = np.zeros(X.shape[0],)
        # print(label.shape)
        mat_tensor = self.get_matrix(label)
        # triangle = self.get_triangle()
        self.mat_tensor = mat_tensor
        result = self.model.predict((X, label, mat_tensor, event))#, triangle))
        result = result[0]
        # print(result)
        return result[:, :-1]


    def predict_survival_func(self, X):
        score = self.predict_score(X)
        # print(score.shape)
        return 1-np.cumsum(pd.DataFrame(score).T, axis=0)

    def concordance_eval(self, X = None, event_times = None, event_observed = None):
        if(type(event_times) == type(None) and type(X) == type(None) and type(event_observed) == type(None)):
            event_times = self.label
            predicted_scores = (self.predict_survival_func(self.X)).sum(axis = 0)
            event_observed = self.event
        else:
            predicted_scores = (self.predict_survival_func(X)).sum(axis=0)
        # print(event_times.shape, predicted_scores)
        index = concordance_index(event_times, predicted_scores, event_observed)
        print("concordance index value is: ", index)
        logger.info('concordance index is {}'.format(index))
        return index

    def plot_survival_func(self, X, obj):
        surv_df = self.predict_survival_func(X)
        get_survival_plot(surv_df, obj)




## loss layer
class Total_Loss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # self.alpha = alpha
        # self.beta = beta
        super(Total_Loss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y_true, nll_mat, event, y_pred = inputs
        # tf.print(y_pred)
        # if sum(sum(np.isnan(nll_mat)))!=0:
        #    print("break")
        # nll loss
        # tf.print(nll_mat.get_shape)
        tmp = tf.reduce_sum((nll_mat * tf.cast(tf.reshape(y_pred, [-1, MAX_MAT_COL+1]), dtype=tf.float32)), axis=1,
                            keepdims=True)
        # if sum(sum(np.isnan(tmp)))!=0:
        #    print("break")
        # tf.print(tmp)

        log_likelihood_loss = - tf.reduce_mean(tf.math.log(tmp))
        # tf.print(log_likelihood_loss)
        # tf.print("break")


        total_loss = log_likelihood_loss

        self.add_loss(total_loss, inputs=True)
        self.add_metric(total_loss, aggregation="mean", name="nll_loss")

        return total_loss

## transform layer
class Transform(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # self.alpha = alpha
        # self.beta = beta
        super(Transform, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y_pred, triangle = inputs


        prediction = tf.matmul(y_pred, triangle)

        # tf.print(prediction.shape)
        # tf.print('break')
        temp = tf.exp(prediction)
        # tf.print(temp.shape)
        Z = tf.reduce_sum(temp, axis=1, keepdims=True)
        # tf.print(Z.shape)
        result = tf.divide(temp, Z)

        return result















