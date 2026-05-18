import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

n, d= 500, 8
true_weight= np.random.randn(d)
X= np.random.randn(n, d)
y= X @ true_weight + 2.0 * np.random.randn(n)

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

# STABILITY 

def stability(X_train, y_train, X_test, lam):
    n = len(X_train)

    w_all = ridge_regression(X_train, y_train, lam)
    predict_all = predict(X_test, w_all)
    total_difference = 0.0

    for i in range(n):
        X_loo = np.delete(X_train, i, axis=0) # leave one out
        y_loo = np.delete(y_train, i)

        w_loo = ridge_regression(X_loo, y_loo, lam) # train again like that
        predict_loo = predict(X_test, w_loo)
        
        difference = np.mean(np.abs(predict_all - predict_loo))
        total_difference += difference

    return total_difference / n

for lam in [0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]: #U shape for synthetic data?
    Stability = stability(X_train, y_train, X_test, lam)
    print(f"λ: {lam:6}, Stability: {Stability:.6f}")  

for lam in [0, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
    Stability = stability(X_train_conc_norm, y_train_conc_norm, X_test_conc_norm, lam)
    print(f"λ: {lam:6}, Stability: {Stability:.6f}")  

# FULL EXPERIMENT

lambda_synthetic = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 
               10.0, 20.0, 50.0, 100.0]
lambda_concrete = [0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 
                200.0, 500.0, 1000.0, 5000.0, 10000.0]

# result lists
results_synthetic = {
    'train_error': [],
    'test_error': [],
    'stability': []
}

results_concrete = {
    'train_error': [],
    'test_error': [],
    'stability': []
}

for lam in lambda_synthetic:
    w = ridge_regression(X_train, y_train, lam)
    results_synthetic['train_error'].append(mse(y_train, predict(X_train, w)))
    results_synthetic['test_error'].append(mse(y_test, predict(X_test, w)))
    results_synthetic['stability'].append(stability(X_train, y_train, X_test, lam))

for lam in lambda_concrete:
    w_conc = ridge_regression(X_train_conc_norm, y_train_conc_norm, lam)
    results_concrete['train_error'].append(mse(y_train_conc_norm, predict(X_train_conc_norm, w_conc)))
    results_concrete['test_error'].append(mse(y_test_conc_norm, predict(X_test_conc_norm, w_conc)))
    results_concrete['stability'].append(stability(X_train_conc_norm, y_train_conc_norm, X_test_conc_norm, lam))

print("Done!")

"""
print("SENTETİK ")
for i, lam in enumerate(lambda_synthetic):
    print(f"λ={lam:6} | Train: {results_synthetic['train_error'][i]:.4f} | "
          f"Test: {results_synthetic['test_error'][i]:.4f} | "
          f"Stab: {results_synthetic['stability'][i]:.6f}")

print("\nCONCRETE")
for i, lam in enumerate(lambda_concrete):
    print(f"λ={lam:7} | Train: {results_concrete['train_error'][i]:.4f} | "
          f"Test: {results_concrete['test_error'][i]:.4f} | "
          f"Stab: {results_concrete['stability'][i]:.6f}")

"""

# GRAPHS

# Synthetic

plt.figure(figsize=(8, 5))
plt.plot(lambda_synthetic, results_synthetic['train_error'], marker='o', 
         label='Train MSE', color= 'purple')
plt.plot(lambda_synthetic, results_synthetic['test_error'], marker='o', 
         label='Test MSE', color= 'blue')
plt.xscale('symlog') #we have λ = 0
plt.xlabel('λ')
plt.ylabel('MSE')   
plt.title('Synthetic Error vs λ')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(lambda_synthetic, results_synthetic['stability'], marker='o', 
         label='Stability', color= 'pink')
plt.xscale('symlog') #we have λ = 0
plt.xlabel('λ')
plt.ylabel('Stability')   
plt.title('Synthetic Stability vs λ')
plt.legend()
plt.grid(True)
plt.show()

# Concrete

plt.figure(figsize=(8, 5))
plt.plot(lambda_concrete, results_concrete['train_error'], marker='o', 
         label='Train MSE', color= 'purple')
plt.plot(lambda_concrete, results_concrete['test_error'], marker='o', 
         label='Test MSE', color= 'blue')
plt.xscale('symlog') #we have λ = 0
plt.xlabel('λ')
plt.ylabel('MSE')   
plt.title('Concrete Error vs λ')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(lambda_concrete, results_concrete['stability'], marker='o', 
         label='Stability', color= 'pink')
plt.xscale('symlog') #we have λ = 0
plt.xlabel('λ')
plt.ylabel('Stability')   
plt.title('Concrete Stability vs λ')
plt.legend()
plt.grid(True)
plt.show()

# more stable models tend to generalize better?

# Synthetic
cutoff = 8 # λ ≤ 20
plt.figure(figsize=(8, 5))
plt.scatter(results_synthetic['stability'][:cutoff], results_synthetic['test_error'][:cutoff],
            color='purple', zorder= 5)

for i, lam in enumerate(lambda_synthetic):
    plt.annotate(f"λ={lam}", (results_synthetic['stability'][i], 
                              results_synthetic['test_error'][i]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)
plt.xlabel('Stability')
plt.ylabel('Test MSE')
plt.title('Synthetic Test Error vs Stability (λ ≤ 20)')
plt.grid(True)
plt.show()

# Concrete

plt.figure(figsize=(8, 5))
plt.scatter(results_concrete['stability'], results_concrete['test_error'],
            color='purple', zorder= 5)

for i, lam in enumerate(lambda_concrete):
    plt.annotate(f"λ={lam}", (results_concrete['stability'][i], 
                              results_concrete['test_error'][i]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)
plt.xlabel('Stability')
plt.ylabel('Test MSE')
plt.title('Concrete Test Error vs Stability')
plt.grid(True)
plt.show()