import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# hands on survival func
def get_survival_func(df, duration, event, log_partial_hazard):
    df['log_partial_hazard'] = log_partial_hazard
    df['partial_hazard'] = np.exp(log_partial_hazard)

    data = df.groupby(duration, as_index=False).agg({event: 'sum', 'partial_hazard': "sum"}).sort_values(duration,
                                                                                                         ascending=False)
    data['cnt'] = len(df)
    data['cum_partial_har'] = data.partial_hazard.cumsum()

    # base line hazard
    data['base_haz'] = 1 - np.exp(data[event] * (-1.0) / data['cum_partial_har'])
    data['base_cumsum_haz'] = data['base_haz'][::-1].cumsum()

    # cumulative hazard
    cum_haz_df = pd.DataFrame(
        np.matrix(data.sort_values(duration, ascending=True).base_cumsum_haz).T * np.matrix(df['partial_hazard']),
        index=data[duration][::-1], columns=list(range(df.shape[0])))

    surv_df = np.exp(-cum_haz_df)

    return surv_df


"""
def get_survival_func(df, duration, event, log_partial_hazard):
    df['log_partial_hazard'] = log_partial_hazard
    df['partial_hazard'] = np.exp(log_partial_hazard)

    data = df.groupby(duration, as_index=False).agg({event: 'sum', 'partial_hazard': "sum"}).sort_values(duration,
                                                                                                         ascending=False)
    data['cnt'] = len(df)
    data['cum_partial_har'] = data.partial_hazard.cumsum()

    # base line hazard
    data['base_haz'] = 1 - np.exp(data[event] * (-1.0) / data['cum_partial_har'])
    # data['base_cumsum_haz'] = data['base_haz'][::-1].cumsum()
    # print(data['base_haz'])
    # print("___________")
    # print(data['base_cumsum_haz'])
    # print(df['partial_hazard'])
    # cumulative hazard
    cum_haz_df = pd.DataFrame(
        np.matrix(data.sort_values(duration, ascending=True).base_haz).T * np.matrix(df['partial_hazard']),
        index=data[duration][::-1], columns=list(range(df.shape[0])))

    # surv_df = np.exp(-cum_haz_df)
    surv_df = (1 - cum_haz_df).cumprod()

    return surv_df

"""


def get_survival_plot(surv_df, obj):
    surv_df[obj].plot(ls="--", color="#A60628", label='survival function')
    plt.legend()
    plt.xlabel("time length")
    plt.ylim(0, 1.1)
    plt.ylabel("probability")
    plt.show()
    return None


def get_cond_survival_func(surv_df, obj, dataset, duration = 'week', plot = True):
    conditioned_sf = surv_df.apply(lambda c: (c / c.loc[dataset.loc[c.name, duration]]).clip(upper=1))
    if plot == True:
        surv_df[obj].plot(ls="--", color="#A60628", label='survival function')
        conditioned_sf[obj].plot(color="#A60628", label='conditioned survival function')
        plt.legend()
        plt.ylim(0, 1.1)
        plt.show()
    return conditioned_sf


