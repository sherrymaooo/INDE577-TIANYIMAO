import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats


class K_Nearest_Neighbor():
    def __init__(self, k=3, classification=True) -> None:
        self.k = k
        self.train_input = None
        self.train_size = None
        self.train_output = None
        self.classification = classification

    def fit(self, train_input, train_output):
        '''
        A knn classifier and regressor.
        
        Input:
            train_input: ndarray, a n*p matrix, which is the features of training data set.
            train_output: ndarray, a n*1 matrix, which is the output (either a class or a value) of training data set.
            new_point: ndarray, a 1*p matrix, which is the input of a new data.
            k: int, number of neighbors.
            classifier: an indicator of classifier and regressor. Default is True.
        
        Output:
            knn_result: int or float, the result of knn prediction.
        '''
        train_input = train_input.to_numpy()
        train_output = train_output.to_numpy()
        self.train_input = train_input
        self.train_size = train_input.shape[0]
        self.train_output = train_output

    
    def predict(self, new_point):
        n = self.train_size
        distance = np.zeros(n)
        
        # Distance function
        for i in range(n):
            distance[i] = np.linalg.norm(self.train_input[i,:] - new_point)
        
        top_k_neighbors = np.argsort(distance)[:self.k]
        k_neighbor_output = self.train_output[top_k_neighbors]
        
        # for classifier
        if self.classification:
            voters = Counter(k_neighbor_output)
            knn_result = voters.most_common(1)[0][0]
        
        # for regressor
        else:
            knn_result = np.mean(k_neighbor_output)
        
        return knn_result


class Linear_Regression():
    def __init__(self, bias=True):
        self.X = None
        self.variables = None
        self.y = None
        self.predictor = None
        self.n = None
        self.p = None
        self.bias = bias
        self.beta_hat = None
        self.y_hat = None

    # model fitting
    def fit(self, X, y):
        self.variables = X.columns
        self.predictor = y.name
        
        X = X.to_numpy()
        y = y.to_numpy()

        if self.bias:
            ones_column = np.ones((X.shape[0], 1))
            X = np.append(ones_column, X, axis=1)

        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1]

        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        self.beta_hat = beta_hat
        self.y_hat = X @ beta_hat

    # predict new data
    def predict(self, x):
        if self.bias:
            ones_column = np.ones((x.shape[0], 1))
            x = np.append(ones_column, x, axis=1)
        return x @ self.beta_hat

    # function of sum of squared errors
    def SSE(self):
        return (self.y-self.y_hat).T@(self.y-self.y_hat)
    
    # function of mean squared errors
    def MSE(self):
        return self.SSE()/(self.n-self.p)

    # function of sum of squares regression
    def SSR(self):
        return (self.y_hat - np.mean(self.y)).T @ (self.y_hat - np.mean(self.y))

    # function of mean squared regression
    def MSR(self):
        return self.SSR()/(self.p-1)

    # function of sum of squares total
    def SST(self):
        return (self.y-np.mean(self.y_hat)).T@(self.y-np.mean(self.y_hat))

    # function of coefficient of determination
    def R_2(self):
        return 1 - self.SSE()/self.SST()

    # function of coefficient of determination on test data
    def pred_R_2(self, input, true):
        pred = self.predict(input)
        pred_SSE = (true-pred).T@(true-pred)
        pred_SST = (true-np.mean(pred)).T@(true-np.mean(pred))
        return 1 - pred_SSE/pred_SST

    # function of adjusted coefficient of determination
    def adj_R_2(self):
        return 1- (1-self.R_2())*(self.n-1)/(self.n-self.p-1)

    # function of standard deviation of coefficients
    def sd_coef(self):
        return np.sqrt(np.diagonal(self.MSE() * np.linalg.inv(self.X.T @ self.X)))

    # function of t statistic and p-value
    def t_stat(self):
        t = self.beta_hat / self.sd_coef()
        t_p = [2*(1-stats.t.cdf(np.abs(i), (self.n-self.p-1))) for i in t]
        return t, t_p

    # function of F statistic and p-value
    def F_stat(self):
        F = self.MSR()/self.MSE()
        df_1 = self.p - 1
        df_2 = self.n - self.p
        #find p-value of F test statistic 
        F_p = 1-stats.f.cdf(F, df_1, df_2) 
        return F, F_p

    # function of root mean square error
    def RMSE(self, input, true):
        pred = self.predict(input)
        return np.sqrt((true-pred).T@(true-pred)/len(true))

    # function of model summary
    def summary(self):
        coef_df = pd.DataFrame()
        
        coef_df['Estimate'] = self.beta_hat
        coef_df['Std.Error'] = self.sd_coef()
        coef_df['t value'] = self.t_stat()[0]
        coef_df['Pr(>|t|)'] = self.t_stat()[1]
        coef_df.index = ['Intercept'] + list(self.variables)

        print(coef_df)

        print(f"Residual standard error: {round(np.sqrt(self.MSE()), 3)} on {self.n-self.p} degress of freedom.")
        print(f"R-squared: {round(self.R_2(), 3)}, Adjusted R-square: {round(self.adj_R_2(), 3)}")
        f_stat_str = f"F-statistic: {round(self.F_stat()[0], 3)} on {self.p - 1} and {self.n - self.p} DF,"
        f_p_str = f"p-value: {round(self.F_stat()[1], 3)}"
        print(f_stat_str + f_p_str)


class Logistic_Regression():
    # initialize
    def __init__(self, bias=True, gamma=0.01, max_iter=100, eta=0.001):
        self.X = None
        self.variables = None
        self.y = None
        self.predictor = None
        self.n = None
        self.p = None
        self.bias = bias
        self.gamma = gamma
        self.max_iter = max_iter
        self.eta = eta

        self.weights = None
        self.weights_history = []
        self.loss_history = [np.inf]
    
    # cross entropy loss of one data
    def cross_entropy_loss(self, y, y_hat):
        return -y*np.log(y_hat) - (1.0-y)*np.log(1.0-y_hat)

    # total cross entropy loss
    def loss(self):
        total_loss = sum(self.cross_entropy_loss(self.y[i], self.sigmoid(x@self.weights)) for i, x in enumerate(self.X))
        return total_loss

    # sigmoid function
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    # gradient of loss
    def gradient_L(self):
        sigmoids = np.array([self.sigmoid(x@self.weights) - self.y[i] for i, x in enumerate(self.X)])
        d_w = sigmoids @ self.X
        return d_w

    # model fitting
    def fit(self, X, y):
        self.variables = X.columns
        self.predictor = y.name
        
        X = X.to_numpy()
        y = y.to_numpy()
        if self.bias:
            ones_column = np.ones((X.shape[0], 1))
            X = np.append(ones_column, X, axis=1)
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1]
        
        weights = np.random.rand(self.p)
        self.weights = weights
        self.weights_history.append(weights)
        for i in range(1, self.max_iter+1):
            dw = self.gradient_L()
            weights = weights - self.gamma * dw
            self.weights = weights
            self.weights_history.append(weights)
            L = self.loss()
            self.loss_history.append(L)
            if i >= self.max_iter or abs(L - self.loss_history[i-1]) <= self.eta:
                break
    
    # predict new data
    def prediction(self, X, weights):
        X = X.to_numpy()
        if self.bias:
            ones_column = np.ones((X.shape[0], 1))
            X = np.append(ones_column, X, axis=1)
        labels = np.array([1, 0])
        y_hat = [self.sigmoid(x @ weights) for x in X]
        return [np.random.choice(labels, p = [y_hat_i, 1.0-y_hat_i]) for y_hat_i in y_hat]


class MultilayerPerceptron():
  
    def __init__(self, layers = [784, 60, 60, 10], actFun_type='relu'):
        self.actFun_type = actFun_type
        self.layers = layers
        self.L = len(self.layers)
        self.W =[[0.0]]
        self.B = [[0.0]]
        for i in range(1, self.L):
            w_temp = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2/self.layers[i-1])
            b_temp = np.random.randn(self.layers[i], 1) * np.sqrt(2/self.layers[i-1])

            self.W.append(w_temp)
            self.B.append(b_temp)

    def reset_weights(self, layers = [784, 60, 60, 10]):
        self.layers = layers
        self.L = len(self.layers)
        self.W = [[0.0]]
        self.B = [[0.0]]
        for i in range(1, self.L):
            w_temp = np.random.randn(self.layers[i], self.layers[i-1])*np.sqrt(2/self.layers[i-1])
            b_temp = np.random.randn(self.layers[i], 1)*np.sqrt(2/self.layers[i-1])

            self.W.append(w_temp)
            self.B.append(b_temp)

    def forward_pass(self, p, predict_vector = False):
        Z =[[0.0]]
        A = [p[0]]
        for i in range(1, self.L):
            z = (self.W[i] @ A[i-1]) + self.B[i]
            a = self.actFun(z, self.actFun_type)
            Z.append(z)
            A.append(a)

        if predict_vector == True:
            return A[-1]
        else:
            return Z, A

    def mse(self, a, y):
        return .5*sum((a[i]-y[i])**2 for i in range(10))[0]

    def MSE(self, data):
        c = 0.0
        for p in data:
            a = self.forward_pass(p, predict_vector=True)
            c += self.mse(a, p[1])
        return c/len(data)

    def actFun(self, z, type):
        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        elif type == 'relu':
            return np.maximum(0, z)
        else:
            return None

    def diff_actFun(self, z, type):
        if type == 'tanh':
            return 1.0 - (np.tanh(z))**2
        elif type == 'sigmoid':
            return self.actFun(z, type) * (1-self.actFun(z, type))
        elif type == 'relu':
            return np.where(z > 0, 1.0, 0)
        else:
            return None

    def deltas_dict(self, p):
        Z, A = self.forward_pass(p)
        deltas = dict()
        deltas[self.L-1] = (A[-1] - p[1])*self.diff_actFun(Z[-1], self.actFun_type)
        for l in range(self.L-2, 0, -1):
            deltas[l] = (self.W[l+1].T @ deltas[l+1]) *self.diff_actFun(Z[l], self.actFun_type)

        return A, deltas

    def stochastic_gradient_descent(self, data, alpha = 0.04, epochs = 3):
        print(f"Initial Cost = {self.MSE(data)}")
        for k in range(epochs):
            for p in data:
                A, deltas = self.deltas_dict(p)
                for i in range(1, self.L):
                    self.W[i] = self.W[i] - alpha*deltas[i]@A[i-1].T
                    self.B[i] = self.B[i] - alpha*deltas[i]
        print(f"{k} Cost = {self.MSE(data)}")


    def mini_batch_gradient_descent(self, data, batch_size = 15, alpha = 0.04, epochs = 3):
        print(f"Initial Cost = {self.MSE(data)}")
        data_length = len(data)
        for k in range(epochs):
            for j in range(0, data_length-batch_size, batch_size):
                delta_list = []
                A_list = []
                for p in data[j:j+batch_size]:
                    A, deltas = self.deltas_dict(p)
                    delta_list.append(deltas)
                    A_list.append(A)

                for i in range(1, self.L):
                    self.W[i] = self.W[i] - (alpha/batch_size)*sum(da[0][i]@da[1][i-1].T for da in zip(delta_list, A_list))
                    self.B[i] = self.B[i] - (alpha/batch_size)*sum(deltas[i] for deltas in delta_list)
            print(f"{k} Cost = {self.MSE(data)}")


class Perceptron():
    # initialize
    def __init__(self, bias=True, gamma=0.01, max_iter=100, eta=0.001) -> None:
        self.X = None
        self.variables = None
        self.y = None
        self.predictor = None
        self.n = None
        self.p = None
        self.bias = bias
        self.gamma = gamma
        self.max_iter = max_iter
        self.eta = eta

        self.weights = None
        self.weights_history = []
        self.loss_history = [np.inf]

    def sign(self, x, y):
        if x@y>0:
            return 1
        else:
            return -1
        
    def loss(self):
        return sum(0.5*(self.sign(self.weights, x) - self.y[i])**2 for i, x in enumerate(self.X))

    def grad_approx(self, x, y):
        return (self.sign(self.weights, x) - y) * x

    def update_w(self, id):
        self.weights = self.weights - self.gamma * self.grad_approx(self.X[id], self.y[id])
        self.weights_history.append(self.weights)
        return 

    # model fitting
    def fit(self, X, y):
        self.variables = X.columns
        self.predictor = y.name
        
        X = X.to_numpy()
        y = y.to_numpy()
        if self.bias:
            ones_column = np.ones((X.shape[0], 1))
            X = np.append(ones_column, X, axis=1)
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1]
        
        weights = np.random.rand(self.p)
        self.weights = weights
        self.weights_history.append(weights)

        for i in range(1, self.max_iter+1):
            random_id = np.random.randint(self.n)
            weights = self.update_w(random_id)
            L = self.loss()
            self.loss_history.append(L)
            if i >= self.max_iter:
                break

    # predict new data
    def prediction(self, X, weights):
        X = X.to_numpy()
        if self.bias:
            ones_column = np.ones((X.shape[0], 1))
            X = np.append(ones_column, X, axis=1)
        y_hat = [self.sign(x, weights) for x in X]
        return y_hat