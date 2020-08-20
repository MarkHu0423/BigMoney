
# %% 导入必要的库
from math import sin
import backtrader as bt
import matplotlib.pyplot as plt
import akshare as ak
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import math

from tensorflow import keras

from joblib import dump, load
from datetime import datetime

# %% 定义调用预测模型函数
def multi_step_predict(model,data,N=1,feature_size=1):
    y_hat=[]
    x_test_last=data
    for i in range(N):
        y_hat_s1=model.predict(x_test_last.reshape(1,-1,feature_size))[0,0]
        x_test_last=np.roll(x_test_last,-1)
        x_test_last[-1]=y_hat_s1
        y_hat.append(y_hat_s1)
    return y_hat

# %% 定义主策略程序

class MyStrategy(bt.Strategy):
    params = (("maperiod", 20),) 
    model=0
    x_scalar=0
    y_scalar=0

    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.buy_price = None
        self.buy_comm = None

        self.win_size=50
        self.feature_size=4
        self.x_last=np.zeros((self.win_size,self.feature_size))
        self.days_elapsed=0
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod) # 添加移动均线指标

    def next(self):
        future=1 # 设置预测天数
        self.days_elapsed+=1
        self.x_last=np.roll(self.x_last,-1)
        self.x_last[-1]=np.array([self.data.open[0],self.data.high[0],self.data.low[0],self.data.close][0])


        if self.days_elapsed<self.win_size:
            return
        
        scaled_x=x_scalar.transform(self.x_last)
        pred_close=multi_step_predict(model,scaled_x,N=future,feature_size=self.feature_size)

        twenty_avg_lines=y_scalar.transform(np.array(self.sma[0]).reshape(1,1))[0,0] #得到20日均线
        if self.order:  # 检查是否有指令等待执行,
            return
        # 检查是否持仓
        if not self.position:  # 没有持仓
            if pred_close > twenty_avg_lines:  # 执行买入条件判断：预测未来一天收盘价格上涨突破20日均线
                self.order = self.buy(size=1000)  # 执行买入1000股
        else:
            if pred_close < twenty_avg_lines:  # 执行卖出条件判断：预测未来一天收盘价格跌破20日均线
                self.order = self.sell(size=1000)  # 执行卖出1000股


# %% 初始化策略
model = keras.models.load_model('stock_model.mod')
x_scalar=load('x_scaler.bin')
y_scalar=load('y_scaler.bin')

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

stock_hfq_df = ak.stock_zh_a_daily(symbol="sh600000", adjust="hfq")  # 利用 AkShare 获取后复权数据
# %% 加载预测模型
MyStrategy.model=model
MyStrategy.x_scalar=x_scalar
MyStrategy.y_scalar=y_scalar


# %% 回测系统设置
cerebro = bt.Cerebro()  # 初始化回测系统
start_date = datetime(2018, 7, 1)  # 回测开始时间
end_date = datetime(2020, 7, 1)  # 回测结束时间
data = bt.feeds.PandasData(dataname=stock_hfq_df, fromdate=start_date, todate=end_date)  # 加载数据
cerebro.adddata(data)  # 将数据传入回测系统
cerebro.addstrategy(MyStrategy)  # 将交易策略加载到回测系统中
start_cash = 1000000
cerebro.broker.setcash(start_cash)  # 设置初始资本为 100000
cerebro.broker.setcommission(commission=0.002)  # 设置交易手续费为 0.2%
cerebro.run()  # 运行回测系统

port_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
pnl = port_value - start_cash  # 盈亏统计
print(f"初始资金: {start_cash}\n回测期间：{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}")
print(f"总资金: {round(port_value, 2)}")
print(f"净收益: {round(pnl, 2)}")

