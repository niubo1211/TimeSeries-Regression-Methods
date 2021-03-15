import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import AR
from statsmodels.tsa.api import AutoReg
from statsmodels.tsa.api import ARMA
from statsmodels.tsa.api import ARIMA
from greytheory import GreyTheory
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.formula.api import mixedlm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor



class ts():

    def __init__(self):
        self.data = pd.read_excel('C:/Users/user/Desktop/测试数据.xlsx')
        self.data = self.data.dropna(axis = 0, subset = ['WindSpeed', 'slipRate'])

    def get_data(self):
        return self.data

    def train_test_split(self, data, y_label: str, train_ratio: float):
        train_size = round(len(data) * train_ratio)
        self.x_train = data[:train_size]
        self.x_test = data[train_size:]
        self.y_train = data[:train_size][y_label]
        self.y_test = data[train_size:][y_label]
        return self.x_train, self.x_test, self.y_train, self.y_test

    def plot_ts(self, data, x_label, y_label, color_type):
        plt.figure()
        plot = sns.lineplot(x=x_label, y=y_label, data=data, color=color_type)
        return plot

    def get_accuracy(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def show_result(self, x_train, y_train, x_test, y_test, y_pred):
        fig, ax = plt.subplots()
        ax.plot(x_train['timestamp'], y_train, color='blue')
        ax.plot(x_test['timestamp'], y_test, color='green')
        ax.plot(x_test['timestamp'], y_pred, color='red')
        plt.show()

    # naive forecast
    # y^t+i = yt based on last value of train set
    def naive_forecast(self):
        y_hat = self.x_test.copy()
        y_hat['naive'] = self.y_train[len(self.y_train) - 1]
        y_pred = y_hat['naive']
        return y_pred

    # simple average
    def simple_average(self):
        y_hat = self.x_test.copy()
        y_hat['simple_avg'] = self.y_train.mean()
        y_pred = y_hat['simple_avg']
        return y_pred

    # moving average
    def moving_average(self, window_size: int):
        y_hat = self.x_test.copy()
        y_hat['moving_avg'] = self.y_train.rolling(window_size).mean().iloc[-1]
        y_pred = y_hat['moving_avg']
        return y_pred

    # simple exponential smoothing
    def simple_exp_smooth(self, smoothing_level: float):
        y_hat = self.x_test.copy()
        model = SimpleExpSmoothing(np.asarray(self.y_train)).fit(smoothing_level=smoothing_level, optimized=False)
        y_hat['ses'] = model.forecast(len(self.y_test))
        y_pred = y_hat['ses']
        return y_pred

    # Holt's linear trend
    def Holt_linear_trend(self, smoothing_level: float, smoothing_slope: float):
        y_hat = self.x_test.copy()
        model = Holt(np.asarray(self.y_train)).fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope)
        y_hat['hlt'] = model.forecast(len(self.y_test))
        y_pred = y_hat['hlt']
        return y_pred

    # Holt-winter
    def Holt_winter(self, seasonal_periods: int, trend: str, seasonal: str):
        y_hat = self.x_test.copy()
        model = ExponentialSmoothing(np.asarray(self.y_train), seasonal_periods=seasonal_periods, trend=trend,
                                     seasonal=seasonal).fit()
        y_hat['hw'] = model.forecast(len(self.y_test))
        y_pred = y_hat['hw']
        return y_pred

    # AR
    def AR(self):
        y_hat = self.x_test.copy()
        model = AR(self.y_train).fit()
        y_hat['ar'] = model.predict(start=len(self.y_train), end=len(self.y_train) + len(self.y_test) - 1,
                                    dynamic=False)
        y_pred = y_hat['ar']
        return y_pred

    # AutoReg
    # have to decide p lags first
    def AutoReg(self, p: int):
        y_hat = self.x_test.copy()
        model = AutoReg(self.y_train, lags=p).fit()
        y_hat['ar'] = model.predict(start=len(self.y_train), end=len(self.y_train) + len(self.y_test) - 1)
        y_pred = y_hat['ar']
        return y_pred

    # MA
    # have to decide q lags
    def MA(self, q: int):
        y_hat = self.x_test.copy()
        model = ARMA(self.y_train, order=(0, q)).fit()
        y_hat['ma'] = model.predict(start=len(self.y_train), end=len(self.y_train) + len(self.y_test) - 1)
        y_pred = y_hat['ma']
        return y_pred

    # ARMA
    # have to decide p, q
    def ARMA(self, p: int, q: int):
        y_hat = self.x_test.copy()
        model = ARMA(self.y_train, order=(p, q)).fit()
        y_hat['arma'] = model.predict(start=len(self.y_train), end=len(self.y_train) + len(self.y_test) - 1)
        y_pred = y_hat['arma']
        return y_pred

    # ARIMA
    # have to decide p, i, q
    def ARIMA(self, p: int, i: int, q: int):
        y_hat = self.x_test.copy()
        model = ARIMA(self.y_train, order=(p, i, q)).fit()
        y_hat['arima'] = model.predict(start=len(self.y_train), end=len(self.y_train) + len(self.y_test) - 1)
        y_pred = y_hat['arima']
        return y_pred

    # Croston
    def Croston(self, data, extra_periods=1, alpha=0.1):
        d = np.array(data)  # transform the input into a npumpy array
        cols = len(d)  # historical period length
        d = np.append(d, [np.nan] * extra_periods)  # append np.nan into the demand array to cols

        a, p, f = np.full((3, cols + extra_periods), np.nan)
        q = 1 #periods since last demand observation

        #Initialization
        first_occurence = np.argmax(d[:cols]>0)
        a[0] = d[first_occurence]
        p[0] = 1 + first_occurence
        f[0] = a[0]/p[0]
        #create all the t+1 forecasts
        for t in range(0,cols):
            if d[t] > 0:
                a[t+1] = alpha*d[t] + (1-alpha)*a[t]
                p[t+1] = alpha*q + (1-alpha)*p[t]
                f[t+1] = a[t+1]/p[t+1]
                q = 1
            else:
                a[t+1] = a[t]
                p[t+1] = p[t]
                f[t+1] = f[t]
                q += 1
        #future forecast
        a[cols+1:cols+extra_periods] = a[cols]
        p[cols+1:cols+extra_periods] = p[cols]
        f[cols+1:cols+extra_periods] = f[cols]

        df = pd.DataFrame.from_dict({'Demand':d, 'Forecast':f, 'Period':p, 'Level':a, 'Error':d-f})
        return df

    # 灰色预测
    def GM(self, data: list):
        grey = GreyTheory()
        gm11 = grey.gm11 #GM11
        gm11.alpha = 0.5 #to try customized alpha for IAGO of Z
        gm11.convolution = True #convolutional forecasting of GM11
        gm11.length = 4 # number of convolutional kernel
        for num in data:
            gm11.add_pattern(num, num)
        gm11.forecast()
        for forecast in gm11.analyzed_results:
            if forecast.tag != gm11._TAG_FORECAST_HISTORY:
                res = forecast.forecast_value
        print("result:", res)

        X1 = np.linspace(0, len(gm11.patterns), len(gm11.patterns), endpoint=True)
        plt.plot(X1, gm11.patterns)

        predict_list = []
        for num in gm11.analyzed_results:
            predict_list.append(num.forecast_value)
        #print("predict_list:", predict_list)
        X2 = np.linspace(0, len(predict_list), len(predict_list), endpoint=True)
        plt.plot(X2, predict_list)
        plt.show()

    # 广义线性模型
    # 3种分布类型：Poisson, Tweedie, Gamma
    def GLM(self, X_train, y_train, X_test, distribution: str):
        if distribution == 'Poisson':
            model = PoissonRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return y_pred
        if distribution == 'Tweedie':
            model = TweedieRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return y_pred
        if distribution == 'Gamma':
            model = GammaRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return y_pred
        else:
            print('No requested distribution type')


    # Dynamic Time Wrapping
    def dtw(self, seq1, seq2):
        m = len(seq1)
        n = len(seq2)
        #初始化一个m行n列的距离矩阵
        distance = np.zeros(shape=(m,n))
        for i in range(m):
            for j in range(n):
                distance[i,j] = (seq1[i] - seq2[j])**2 #计算seq1[i]到seq2[j]之间的距离并保存在距离矩阵中
        #构建一个累计距离矩阵
        D = np.zeros(shape=(m,n))
        D[0,0] = distance[0,0]#边界问题
        for i in range(1,m):
            D[i,0] = distance[i,0] + D[i-1,0]
        for j in range(1,n):
            D[0,j] = distance[0,j] + D[0, j-1]
        for i in range(1,m):
            for j in range(1,n):
                D[i,j] =  distance[i,j] + np.min(D[i-1,j],D[i,j-1],D[i-1,j-1])
        return  D[m-1,n-1]


    # 线性回归

    def LinearRegression(self, X_train, y_train, X_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    # 岭回归
    def RidgeRegression(self, X_train, y_train, X_test, alpha: float):
        model = Ridge(alpha = alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #核岭回归
    def KernelRidge(self, X_train, y_train, X_test, alpha: float, gamma: float):
        model = KernelRidge(kernel= 'rbf', alpha = alpha, gamma = gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #套索回归
    def Lasso(self, X_train, y_train, X_test, alpha: float):
        model = Lasso(alpha = alpha)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #最小角回归
    def Lars(self, X_train, y_train, X_test, n_nonzero_coefs: int):
        model = Lars(n_nonzero_coefs = n_nonzero_coefs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #弹性网回归
    def ElasticNet(self, X_train, y_train, X_test, l1_ratio: float):
        model = ElasticNet(l1_ratio = l1_ratio)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #贝叶斯线性回归


    #贝叶斯岭回归
    def BayesianRidge(self, X_train, y_train, X_test):
        model = BayesianRidge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #多项式回归
    def PolyRegression(self, X_train, y_train, X_test, degree: int):
        PF = PolynomialFeatures(degree = degree)
        X_train = PF.fit_transform(X_train.reshape(-1, 1))
        X_test = PF.fit_transform(X_test.reshape(-1, 1))
        y_train = y_train.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #指数回归
    #y = a*e^bx
    def ExponentialRegression(self, x_train, y_train, x_test):
        popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), x_train, y_train)
        a = popt[0]
        b = popt[1]
        y_pred = [a*np.exp(b*t) for t in x_test]
        return y_pred

    #对数回归
    #y = a + b*log(x)
    def LogarithmicRegression(self, x_train, y_train, x_test):
        popt, pcov = curve_fit(lambda t,a,b: a+b*np.log(t), x_train, y_train)
        a = popt[0]
        b = popt[1]
        y_pred = [a+b*np.log(t) for t in x_test]
        return y_pred

    #分位数回归
    def QuantRegression(self, X_train, y_train, X_test, q: float):
        model = QuantReg(y_train, X_train)
        result = model.fit(q = q)
        y_pred = result.predict(X_test)
        return y_pred

    #随机系数回归
    def RandomCoefRegression(self, X_train, y_train, X_test):
        model = mixedlm()

    #无序回归


    #高斯过程
    def GaussianProcessRegression(self, X_train, y_train, X_test):
        kernel = DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel = kernel,
                                       random_state = 0).fit(X_train, y_train)
        y_pred =  gpr.predict(X_test, return_std = True)
        return y_pred

    #Theil-Sen回归
    def TheilSenRegression(self, X_train, y_train, X_test):
        model = TheilSenRegressor(random_state = 0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #Huber回归
    def HuberRegression(self, X_train, y_train, X_test):
        model = HuberRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    #RANSAC
    def RANSACRegression(self, X_train, y_train, X_test):
        model = RANSACRegressor(random_state = 0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    def show_result(self, x_train, y_train, x_test, y_test, y_pred):
        fig, ax = plt.subplots()
        ax.plot(x_train['timestamp'], y_train, color='blue')
        ax.plot(x_test['timestamp'], y_test, color='green')
        ax.plot(x_test['timestamp'], y_pred, color='red')
        plt.show()




