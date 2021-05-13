import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential

# global MAX_MAT_COL
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



class DeepHit:
    def __init__(self, df, label, event,
                 elements_1 =[32, 64], activation_1 = ['relu', 'tanh'],
                 elements_2 = [64, 64], activation_2 = ['tanh', 'tanh'],
                 X_col=None
                 ):
        # print("using deephit!")
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
        MAX_MAT_COL = int(np.max(self.label) + 2)
        self.MAX_MAT_COL = MAX_MAT_COL

        self.model = self.nn_struct(elements_1, elements_2, activation_1, activation_2)

    def nn_struct(self, elements_1, elements_2 , activation_1, activation_2, type1_loss = 20.0, type2_loss = 1.0):
        inputs = tf.keras.layers.Input([len(self.X[0]), ], name='input_X')
        inputs_label = tf.keras.layers.Input([1, ], name="input_label")
        inputs_nllmat = tf.keras.layers.Input([self.MAX_MAT_COL, ], name="input_nllmat")
        inputs_rankmat = tf.keras.layers.Input([self.MAX_MAT_COL, ], name="input_rankmat")
        inputs_event = tf.keras.layers.Input([1, ], name="input_event")

        # share_h1 = tf.keras.layers.Dense(len(self.X[0]), )(inputs)
        # share_h1 = tf.keras.layers.BatchNormalization()(share_h1)
        share_h2 = tf.keras.layers.Dense(elements_1 [0], activation=activation_1[0], activity_regularizer="l2")(inputs)

        for (i, j) in zip(elements_1[1:], activation_1[1:]):
            share_h2 = tf.keras.layers.BatchNormalization()(share_h2)
            share_h2 = tf.keras.layers.Dense(i, activation=j, activity_regularizer="l2")(share_h2)
            # share_h2 = tf.keras.layers.Dense(i, activation='tanh', activity_regularizer="l2")(share_h2)


        hid = tf.keras.layers.concatenate([inputs, share_h2], axis=1)
        output = tf.keras.layers.Dense(elements_2[0], activation=activation_2[0])(hid)

        for (i, j) in zip(elements_2[1:], activation_2):
            output = tf.keras.layers.BatchNormalization()(output)
            output = tf.keras.layers.Dense(i, activation=j)(output)

        # ----------------------------------------------------------------------------------------------
        # share_h1 = tf.keras.layers.Dense(len(self.X[0]), )(inputs)
        # share_h1 = tf.keras.layers.BatchNormalization()(share_h1)
        # share_h1 = tf.keras.layers.Dense(64, activation='relu', activity_regularizer="l2")(share_h1)
        #
        # share_h2 = tf.keras.layers.Dense(64)(share_h1)
        # share_h2 = tf.keras.layers.BatchNormalization()(share_h2)
        # share_h2 = tf.keras.layers.Dense(96, activation='tanh', activity_regularizer="l2")(share_h2)
        #
        # # share_h3 = tf.keras.layers.Dense(32)(share_h2)
        # # share_h3 = tf.keras.layers.BatchNormalization()(share_h3)
        # # share_h3 = tf.keras.layers.Dense(16, activation='selu', activity_regularizer="l2")(share_h3)
        #
        # # print("inputs", inputs)
        # # print("share", share_h2)
        # hid = tf.keras.layers.concatenate([inputs, share_h2], axis=1)
        #
        # output = tf.keras.layers.Dense(96)(hid
        #                                    )
        # output = tf.keras.layers.BatchNormalization()(output)
        # -----------------------------------------------------------------------------------------------
        output = tf.keras.layers.BatchNormalization()(output)
        outputs = tf.keras.layers.Dense(self.MAX_MAT_COL, activation='softmax', activity_regularizer="l2", name='output')(output)
        # tf.print(outputs.shape)
        # print(self.MAX_MAT_COL)

        my_loss = Total_Loss(type1_loss, type2_loss)([inputs_label, inputs_nllmat, inputs_rankmat, inputs_event, outputs])

        model = tf.keras.models.Model(inputs=[inputs, inputs_label, inputs_nllmat, inputs_rankmat, inputs_event],
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
                       # batch_size=batch_size
                       )

        if summary:
            self.model.summary()

    def get_traindata(self):
        def map_fn(X, label, nll_mat, rank_mat, event):
            inputs = {"input_X": X, "input_label": label, "input_nllmat": nll_mat, "input_rankmat": rank_mat,
                      "input_event": event}
            targets = {}
            return inputs, targets

        #     events = tf.convert_to_tensor(event, dtype=tf.float64)
        mat_tensor, rank_mat_tensor = self.get_matrix()
        # print(isinstance(mat_tensor, tf.Tensor),"__________________")
        dataset = tf.data.Dataset.from_tensor_slices((self.X,
                                                      self.label,
                                                      mat_tensor,
                                                      rank_mat_tensor,
                                                      self.event)).map(map_fn).batch(self.batch_size)
        return dataset

    def get_matrix(self, label = None):
        if type(label) == type(None):
            label = self.label
        rows = len(label)
        mat = np.zeros([rows, MAX_MAT_COL])
        for i in range(rows):
            if self.event[i] != 0:
                mat[i, int(self.label[i])] = 1
            else:
                mat[i, int(self.label[i]) + 1:] = 1
        mat_tensor = (tf.convert_to_tensor(mat))

        ##
        rank_rows = len(label)
        rank_mat = np.zeros([rank_rows, MAX_MAT_COL])
        for i in range(rank_rows):
            rank_mat[i, :int(self.label[i] + 1)] = 1
        rank_mat_tensor = (tf.convert_to_tensor(rank_mat))

        return mat_tensor, rank_mat_tensor

    def predict_score(self, X):
        # print(X.shape)
        # print(self.X.shape)
        # print(self.label.shape)
        label = np.zeros(X.shape[0],)
        event = np.zeros(X.shape[0],)
        # print(label.shape)
        mat_tensor, rank_mat_tensor = self.get_matrix(label)
        self.mat_tensor, self.rank_mat_tensor = mat_tensor, rank_mat_tensor
        result = self.model.predict((X, label, mat_tensor, rank_mat_tensor, event))
        result = result[0]
        # print(result)
        return result[:, :-1]
        # return result


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
        # type(index)111111111
        logger.info('concordance index is {}'.format(index))
        return index

    def plot_survival_func(self, X, obj):
        surv_df = self.predict_survival_func(X)
        get_survival_plot(surv_df, obj)







## loss layer
class Total_Loss(tf.keras.layers.Layer):
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta
        super(Total_Loss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y_true, nll_mat, rank_mat, event, y_pred = inputs
        # if sum(sum(np.isnan(nll_mat)))!=0:
        #    print("break")
        # nll loss
        # tf.print(nll_mat.get_shape)
        tmp = tf.reduce_sum((nll_mat * tf.cast(tf.reshape(y_pred, [-1, MAX_MAT_COL]), dtype=tf.float32)), axis=1,
                            keepdims=True)
        # if sum(sum(np.isnan(tmp)))!=0:
        #    print("break")

        log_likelihood_loss = - tf.reduce_mean(tf.math.log(tmp))
        # print(log_likelihood_loss)
        #    print("break")

        # rank loss
        rank_rows = int(tf.shape(y_true)[0])

        sigma = tf.constant(0.1, dtype=tf.float32)
        one_vector = tf.ones([rank_rows, 1], dtype=tf.float32)

        # 第j个用户用第i个用户risk func的结果
        R2 = tf.matmul(rank_mat, tf.transpose(y_pred))
        diag_R = tf.reshape(tf.linalg.diag_part(R2), [-1, 1])

        # 每行都表示第i个用户用自己risk func的结果
        R1 = tf.matmul(diag_R, tf.transpose(one_vector))
        mu = R1 - R2

        # 两两比较，第i个用户的T < 第j个用户的T时，Tij=1
        T = tf.nn.relu(tf.sign(
            tf.matmul(one_vector, tf.reshape(tf.cast(y_true, dtype=tf.float32), [1, -1])) - tf.matmul(
                tf.cast(tf.reshape(y_true, [-1, 1]), dtype=tf.float32), tf.transpose(one_vector))))

        # 事件发生的用户生效
        rank_eff_mat = tf.linalg.diag(tf.squeeze(event))
        A = tf.matmul(rank_eff_mat, T)

        eta = tf.reduce_mean(A * tf.exp(-mu / sigma), axis=1, keepdims=True)
        log_rank_loss = tf.reduce_sum(eta)
        # print(log_rank_loss)
        total_loss = self.alpha * log_likelihood_loss + self.beta * log_rank_loss

        self.add_loss(total_loss, inputs=True)
        self.add_metric(total_loss, aggregation="mean", name="total_loss")

        return total_loss






