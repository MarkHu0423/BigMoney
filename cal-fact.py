import pandas as pd
from util.StochRSI import StochRSI
from util.KDJ import KDJ
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np

def cal_rsi(df0, period=6):  # 默认周期为6日
    df0['diff'] = df0['close'] - df0['close'].shift(1)  # 用diff储存两天收盘价的差
    df0['diff'].fillna(0, inplace=True)  # 空值填充为0
    df0['up'] = df0['diff']  # diff赋值给up
    df0['down'] = df0['diff']  # diff赋值给down
    df0['up'][df0['up'] < 0] = 0  # 把up中小于0的置零
    df0['down'][df0['down'] > 0] = 0  # 把down中大于0的置零
    df0['avg_up'] = df0['up'].rolling(period).sum() / period  # 计算period天内平均上涨点数
    df0['avg_down'] = abs(df0['down'].rolling(period).sum() / period)  # 计算period天内评价下跌点数
    df0['avg_up'].fillna(0, inplace=True)  # 空值填充为0
    df0['avg_down'].fillna(0, inplace=True)  # 空值填充为0
    df0['rsi'] = 100 - 100 / (1 + (df0['avg_up'] / df0['avg_down']))  # 计算RSI
    return df0  # 返回原DataFrame

def fun3_1():
    """
    计算未来收益
    :return:
    """
    df = pd.read_csv('data/maotai-k2018.7-2020.7-data.csv', index_col=0)
    df['r_1'] = (df['close']-df.shift(1)['close'])/df.shift(1)['close']
    df['r_1'] = df['r_1'].shift(-1)

    df['r_5'] = (df['close'] - df.shift(5)['close']) / df.shift(5)['close']
    df['r_5'] = df['r_5'].shift(-5)

    df['r_10'] = (df['close'] - df.shift(10)['close']) / df.shift(10)['close']
    df['r_10'] = df['r_10'].shift(-10)

    df.dropna()
    df.to_csv('data/raw_factor_maotai-k2018.7-2020.7-data.csv')

# fun3_1()


def fun3_2():
    """
    打图看一下收益
    :return:
    """
    df = pd.read_csv('raw_factor_maotai-k2018.7-2020.7-data.csv', index_col=0)
    y = (df['close'] - df.shift(10)['close'])
    plt.plot(y)
    plt.show()
    # 发现前期数据异常 可能不能用  去掉
    # df = df[:385]
    df = df.reset_index(drop=True)
    df.to_csv('maotai-factor_data7.csv')

# fun3_2()

def fun3_3():
    # 计算因子值
    df = pd.read_csv('data/raw_factor_maotai-k2018.7-2020.7-data2.csv', index_col=0)
    df.rename(columns={"high": "H", "low": "L", 'open': 'O', 'close': 'C', 'date': 'time'}, inplace=True)
    data = df.to_dict(orient="records")
    kdj = KDJ(12,6,3)
    rsi = StochRSI(9,3,3,3)
    for i, kline in enumerate(data):
        kdj.cal_index(kline)
        rsi.cal_index(kline)
        df.loc[i, 'k'] = kdj.K
        df.loc[i, 'rsi'] = rsi.rsi
    df = df[11:-10]
    df = df.reset_index(drop=True)
    # df = cal_rsi(df)

    df.to_csv('data/maotai-factor_data7.csv')

# fun3_3()
def fun3_4():
    # 看因子是否平稳  不平稳就要做处理
    df = pd.read_csv('data/maotai-factor_data7.csv', index_col=0)
    plt.plot(df['k'])
    plt.show()

# fun3_4()

def fun3_5():
    def zscore(x):
        return (x - np.mean(x)) / np.std(x)
    df = pd.read_csv('data/maotai-factor_data7.csv', index_col=0)
    ss = StandardScaler()
    # 这里用到了未来信息，实际情况是要在训练集或验证集上去做标准化
    df['x_1'] = ss.fit_transform(df['k'].values.reshape(-1, 1))
    df['x_2'] = ss.fit_transform(df['rsi'].values.reshape(-1, 1))
    df.to_csv('data/maotai-factor_data7.csv')

# fun3_5()

def fun3_6():
    df = pd.read_csv('maotai-factor_data7.csv', index_col=0)
    print("x1 x2", np.corrcoef(df['x_1'], df['x_2'])[0])
    for column in ['r_1', 'r_5', "r_10"]:
        for factor in ['x_1', 'x_2']:
            print(column, factor, np.corrcoef(df[column], df[factor])[0])
# fun3_3()
fun3_6()

