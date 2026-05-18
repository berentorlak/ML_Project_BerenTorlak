import numpy as np
import pandas as pd

np.random.seed(42)

n, d= 200, 8
true_weight= np.random.randn(d)
X= np.random.randn(n, d)
y= X @ true_weight + 0.5 * np.random.randn(n)

split= int(0.8 * n)
X_train, X_test= X[:split], X[split:]
y_train, y_test= y[:split], y[split:]

# we read the real life data
df = pd.read_excel("Concrete_Data.xls")
X_conc= df.iloc[:, :-1].values
y_conc= df.iloc[:, -1].values

# target variable was not homogeneously distributed, caused train MSE> test MSE
# so we shuffle the data
np.random.seed(42)
indices = np.random.permutation(len(X_conc))
X_conc = X_conc[indices]
y_conc = y_conc[indices]

split_conc = int(0.8 * len(X_conc))
X_train_conc, X_test_conc = X_conc[:split_conc], X_conc[split_conc:]
y_train_conc, y_test_conc = y_conc[:split_conc], y_conc[split_conc:]

# we apply normalization since the features are on different scales
X_mean_c = X_train_conc.mean(axis=0)
X_std_c = X_train_conc.std(axis=0)

X_train_conc_norm = (X_train_conc - X_mean_c) / X_std_c
X_test_conc_norm = (X_test_conc - X_mean_c) / X_std_c

# y normalization
y_mean_c = y_train_conc.mean(axis=0)
y_std_c = y_train_conc.std(axis=0)

y_train_conc_norm = (y_train_conc - y_mean_c) / y_std_c
y_test_conc_norm = (y_test_conc - y_mean_c) / y_std_c

#split control
print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)
print("Concrete dataset training set:", X_train_conc.shape, y_train_conc.shape)      
print("Concrete dataset testing set:", X_test_conc.shape, y_test_conc.shape) 

# Functions for ridge regression
def ridge_regression(X, y, lam): # lam: λ
    n,d = X.shape
    I = np.eye(d) # invertible 
    w = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return w

def predict(X, w):
    return X @ w

def mse (y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

# we test to see on synthetic data
w = ridge_regression(X_train, y_train, lam=1.0)
y_predicted_train = predict(X_train, w)
y_predicted_test = predict(X_test, w)

print("Train MSE:", mse(y_train, y_predicted_train))
print("Test MSE:", mse(y_test, y_predicted_test))

# we check the effect of λ on both synthetic and real data 
for lam in [0, 0.01, 0.1, 1.0, 10.0, 100.0]:
    w= ridge_regression (X_train, y_train, lam=lam)
    train_error_s = mse(y_train, predict(X_train, w))
    test_error_s = mse(y_test, predict(X_test, w))
    print(f"λ: {lam:6}, Train MSE: {train_error_s:.4f}, Test MSE: {test_error_s:.4f}")

for lam in [10, 20, 30, 50, 75, 100, 150, 200]: # 50 looks like the spot
    w_conc = ridge_regression(X_train_conc_norm, y_train_conc_norm, lam=lam)
    train_error_conc = mse(y_train_conc_norm, predict(X_train_conc_norm, w_conc))
    test_error_conc = mse(y_test_conc_norm, predict(X_test_conc_norm, w_conc))
    print(f"λ: {lam:6}, Train MSE (Concrete): {train_error_conc:.4f}, Test MSE (Concrete): {test_error_conc:.4f}")

# double check for the normalization and the other things
print("Normalization parameters:")
print("X_mean:", X_mean_c)
print("X_std:", X_std_c)

print("\nX_train_conc_norm - Shape:", X_train_conc_norm.shape)
print("Mean:", X_train_conc_norm.mean(axis=0))
print("Std:", X_train_conc_norm.std(axis=0))

print("\nX_test_conc_norm - Shape:", X_test_conc_norm.shape)
print("Mean:", X_test_conc_norm.mean(axis=0))
print("Std:", X_test_conc_norm.std(axis=0))


print("\ny normalization parameters:")
print("y_mean:", y_mean_c)
print("y_std:", y_std_c)


print("\ny_train_conc_norm - Shape:", y_train_conc_norm.shape)
print("Mean:", y_train_conc_norm.mean(axis=0))
print("Std:", y_train_conc_norm.std(axis=0))

print("\ny_test_conc_norm - Shape:", y_test_conc_norm.shape)
print("Mean:", y_test_conc_norm.mean(axis=0))
print("Std:", y_test_conc_norm.std(axis=0))

