import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
from survival_function_est import get_survival_func

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')

file_handler = logging.FileHandler('training_log.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)



class DeepSurv:

    def __init__(self, df, label, event, elements=[16, 8], activation = ['relu', 'tanh'], X_col=None):

        self.df = df.sample(frac=1.0)
        y_col = [label, event]
        self.y_col = y_col
        # self.Y = df.get(y_col)
        label = np.array(df[label])
        event = np.array(df[event])
        sort_idx = np.argsort(label)[::-1]
        if X_col != None:
            X = np.array(df[X_col]).astype("float64")
        else:
            X = np.array(df.drop(y_col, axis=1).astype("float64"))
        self.X = X[sort_idx]
        self.label = label[sort_idx]
        self.event = event[sort_idx]
        self.model = self.nn_struct(elements, activation)
        self.get_survival_func_flag = 0


    def nn_struct(self, elements, activation):
        input = tf.keras.layers.Input([len(self.X[0]), ], name='input_X')
        event = tf.keras.layers.Input([1, ], name='input_event')

        # inputs_event = tf.keras.layers.Input([1, ], name="input_event")
        # initializer = tf.keras.initializers.he_normal()
        # acti = tf.keras.layers.LeakyReLU(alpha=0.1)
        inputs = tf.keras.layers.BatchNormalization()(input)
        inputs = Dense(elements[0], activation=activation[0])(inputs)
        for (i, j) in zip(elements[1:], activation[1:]):
            inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = Dense(i, activation=j)(inputs)
            inputs = tf.keras.layers.Dropout(0.1)(inputs)
        outputs = Dense(1, "linear")(input)
        output = tf.keras.layers.BatchNormalization()(outputs)
        my_loss = Total_Loss()([output, event])

        model = tf.keras.models.Model(inputs=[input, event],
                                      outputs=[output, my_loss])
        return model

    '''
    def negative_log_likelihood(self, event):

        def loss(y_true, y_pred):
            partial_hazard = tf.math.exp(y_pred)
            log_cum_partial_hazard = tf.math.log(tf.math.cumsum(partial_hazard))
            event_likelihood = tf.multiply(tf.subtract(y_pred, log_cum_partial_hazard), event)
            neg_likelihood = tf.multiply(-1.0, tf.reduce_sum(event_likelihood))
            return neg_likelihood

        return loss
    '''

    def train(self, epochs, batch_size = 256, initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9, summary = 0):

        self.batch_size = batch_size
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
            # loss=self.negative_log_likelihood(self.event)
            # , metrics=[concordance_metric(event = self.event)]
        )

        data = self.get_traindata()

        self.model.fit(data,
                       # self.X,
                       # self.label,
                       epochs=epochs,
                       # batch_size=batch_size,
                       # callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)],
                       shuffle=False)
        # shuffle false!!!

        if summary:
            self.model.summary()



    def get_traindata(self):
        def map_fn(X, event):#, triangle):
            inputs = {"input_X": X,
                      "input_event": event}

            targets = {}
            return (inputs, targets)

        # events = tf.convert_to_tensor(event, dtype=tf.float64)
        # mat_tensor = self.get_matrix()
        # triangle = self.get_triangle()
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.event
                                                      )).map(map_fn).batch(self.batch_size)
        return dataset



    def predict_score(self, X):
        event = np.zeros(X.shape[0],)
        result = self.model.predict((X,event))
        result = result[0]
        # print(result)
        return result

    def concordance_eval(self, X = None, event_times = None, event_observed = None):
        if(type(event_times) == type(None) and type(X) == type(None) and type(event_observed) == type(None)):
            event_times = self.label
            predicted_scores = -(self.predict_score(self.X))
            event_observed = self.event
        else:
            predicted_scores = -(self.predict_score(X))
        index = concordance_index(event_times, predicted_scores, event_observed)
        print("concordance index value is: ", index)
        logger.info('concordance index is {}'.format(index))

        return index

    def predict_survival_func(self, X):
        log_partial_hazard = self.predict_score(X)
        partial_hazard = np.exp(log_partial_hazard)
        partial_hazard = partial_hazard.reshape(-1,)
        # print(partial_hazard.shape)
        label = self.y_col[0]
        event = self.y_col[1]
        # print(label, event)

        if self.get_survival_func_flag == 0:
            self.df['log_partial_hazard'] = log_partial_hazard
            self.df['partial_hazard'] = partial_hazard

            data = self.df.groupby(label, as_index=False).agg({event: 'sum', 'partial_hazard': "sum"}).sort_values(label,
                                                                                                                 ascending=False)
            data['cnt'] = len(self.df)
            data['cum_partial_har'] = data.partial_hazard.cumsum()

            # base line hazard
            data['base_haz'] = 1 - np.exp(data[event] * (-1.0) / data['cum_partial_har'])
            data['base_cumsum_haz'] = data['base_haz'][::-1].cumsum()
            self.get_survival_func_flag = 1
            self.data = data
        #print(self.data.sort_values(label, ascending=True).base_cumsum_haz)
        #print(((self.data.sort_values(label, ascending=True).base_cumsum_haz).T).shape)
        #print(self.df['partial_hazard'].shape)

        cum_haz_df = pd.DataFrame(
            np.matrix(self.data.sort_values(label, ascending=True).base_cumsum_haz).T * np.matrix(partial_hazard),
            index=self.data[label][::-1], columns=list(range(X.shape[0])))


        surv_df = np.exp(-cum_haz_df)

        # print(min(surv_df.index))
        if min(surv_df.index) != 0:
            temp_label = pd.DataFrame({label:list(range(0,int(min(surv_df.index))))})
            temp = pd.DataFrame(np.tile(surv_df.iloc[0].values,(int(min(surv_df.index)),1)),index=temp_label[label])
            # print('execute')
            surv_df = pd.concat([temp, surv_df], axis=0)
            # surv_df.loc[0:min(surv_df.index)] = surv_df[min(surv_df.index)]
        # print(surv_df.iloc[0])
        return surv_df

    def plot_survival_func(self, X, obj):
        surv_df = self.predict_survival_func(X)
        get_survival_plot(surv_df, obj)

class Total_Loss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # self.alpha = alpha
        # self.beta = beta
        super(Total_Loss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y_pred, event= inputs
        partial_hazard = tf.math.exp(y_pred)
        log_cum_partial_hazard = tf.math.log(tf.math.cumsum(partial_hazard))
        event_likelihood = (y_pred - log_cum_partial_hazard) * event
        neg_likelihood = -1.0 * tf.reduce_sum(event_likelihood)

        self.add_loss(neg_likelihood, inputs=True)
        self.add_metric(neg_likelihood, aggregation="mean", name="nll_loss")

        return neg_likelihood















class concordance_metric(tf.keras.metrics.Metric):
    def __init__(self, name="concordance_metric", event = None):
        super(concordance_metric, self).__init__(name=name)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")
        self.event = event

    def update_state(self, y_true, y_pred, sample_weight=None):
        print(type(y_true), y_true.shape)
        print(y_pred.shape)
        # 在此处改代码
        # 该指标可以计算有多少样本被正确分类为属于给定类：
        # y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        # values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        # y_true = tf.constant(y_true)
        # y_pred = tf.constant(y_pred)
        # y_true = y_true.numpy()
        # y_pred = y_pred.numpy()
        # sess = tf.compat.v1.Session()
        # y_true = y_true.eval(session=sess)
        # y_pred = y_pred.eval(session=sess)
        # y_true = tf.make_ndarray(y_true.op.get_attr('value'))
        # y_pred = tf.make_ndarray(y_pred.op.get_attr('value'))
        values = concordance_index(y_true, y_pred, self.event)
        values = tf.cast(values, "float32")
        print('values', values)
        print('sample_weight', sample_weight)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")

            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)






def data_trans(X):
    return X.astype("float32")