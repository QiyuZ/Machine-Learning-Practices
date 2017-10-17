import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from sklearn.metrics import mean_squared_error

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y = [0.40173564, -0.12671902, 0.55487409, 0.4708342, 0.575, 0.34285828, 0.33246747, 1.00781095, 0.91699119, 0.79823413]
fold_num = 5
subset_size = 2

for i in range(fold_num):
    x_train = np.append(x[:i * subset_size] , x[(i + 1) * subset_size:])
    y_train = np.append(y[:i * subset_size] , y[(i + 1) * subset_size:])
    x_test = x[i * subset_size:][:subset_size]
    y_test = y[i * subset_size:][:subset_size]
    
    op_j = -1
    min_error = 100.0
    for j in range(9):
        total_error = 0.0
        
        for k in range(8):
            x_train_loo = np.append(x_train[:k] , x_train[k+1:])
            y_train_loo = np.append(y_train[:k] , y_train[k+1:])
            x_test_loo = x_train[k]
            y_test_loo = y_train[k]
            
            #print(y_train_loo)
            z = np.polyfit(x_train_loo, y_train_loo, j)
            f = np.poly1d(z)
            mse = (y_test_loo - f(x_test_loo)) * (y_test_loo - f(x_test_loo))
            #mse = mean_squared_error(y_test_loo, f(x_test_loo), multioutput='raw_values')
            total_error += mse
            
        total_error /= 8
        if total_error < min_error:
            min_error = total_error
            op_j = j
    
    z = np.polyfit(x_train, y_train, op_j)
    f = np.poly1d(z)
    mse_f = mean_squared_error(y_test, f(x_test), multioutput='raw_values')
    print("Fold "+str(i+1)+"  optimal m: "+str(op_j)+"  test_error: "+str('%.10f' % mse_f))
