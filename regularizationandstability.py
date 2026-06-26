import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

n, d = 500, 8
true_weight = np.array([1.5, -2.0, 0.8, 0.0, 0.0, 0.0, 1.2, 0.0])
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

for i, lam in enumerate(lambda_synthetic[:cutoff]):
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

# Extension 2: effect of the dataset's size on stability
#Synthetic
sample_size_synthetic = [100, 200, 500, 1000]

result_size_synthetic = {}

for n in sample_size_synthetic:
    np.random.seed(42)
    X_s = np.random.randn(n, 8)
    y_s = X_s @ true_weight + 2.0 * np.random.randn(n)
    
    split = int(0.8 * n)
    X_tr, X_te = X_s[:split], X_s[split:]
    y_tr, y_te = y_s[:split], y_s[split:]
    
    stabilities = []
    for lam in lambda_synthetic:
        stab = stability(X_tr, y_tr, X_te, lam)
        stabilities.append(stab)
    
    result_size_synthetic[n] = stabilities
    print(f"n={n} is done.")

plt.figure(figsize=(8, 5))
for n in sample_size_synthetic:
    plt.plot(lambda_synthetic, result_size_synthetic[n], marker='o', label=f'n={n}')
plt.xscale('symlog')
plt.xlabel('λ')
plt.ylabel('Stability')
plt.title('Stability vs λ (Effect of Dataset Size)')
plt.legend()
plt.grid(True)
plt.show()

# Concrete
sample_size_concrete = [200, 400, 600, 824]  # 824 full train set

result_size_concrete = {}

for n in sample_size_concrete:
    np.random.seed(42) # we choose random points from train set
    indices = np.random.choice(len(X_train_conc_norm), n, replace=False)
    X_c = X_train_conc_norm[indices]
    y_c = y_train_conc_norm[indices]
    
    split = int(0.8 * n)
    X_tr, X_te = X_c[:split], X_c[split:]
    y_tr, y_te = y_c[:split], y_c[split:]
    
    stabilities = []
    for lam in lambda_concrete:
        stab = stability(X_tr, y_tr, X_test_conc_norm, lam)
        stabilities.append(stab)
    
    result_size_concrete[n] = stabilities
    print(f"n={n} is done.")

plt.figure(figsize=(8, 5))
for n in sample_size_concrete:
    plt.plot(lambda_concrete, result_size_concrete[n], marker='o', label=f'n={n}')
plt.xscale('log')
plt.xlabel('λ')
plt.ylabel('Stability')
plt.title('Stability vs λ (Effect of Dataset Size (Concrete))')
plt.legend()
plt.grid(True)
plt.show()


# Extension 3 L1 vs L2

# soft threshold function for Lasso
def soft_thresholding(z, lam):
    if z < -lam:
        return z + lam
    elif z > lam:
        return z - lam
    else:
        return 0.0
    
# since Lasso does not have a closed solution, we will use coordinate descent 
def lasso_coordinate_descent(X, y, lam, max_iteration=1000, tol=1e-4):
    n, d = X.shape
    w = np.zeros(d)
    
    for iteration in range(max_iteration):
        w_pr = w.copy() # we keep the previous weight
        
        for j in range(d):
            y_pred = X @ w
            residual = y - y_pred + X[:, j] * w[j]
            z = np.dot(X[:, j], residual) / n

            w[j] = soft_thresholding(z, lam)

        if np.max(np.abs(w - w_pr)) < tol:
                break
    return w


lambdas_ex3 = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

#Synthetic
results_L2_s = {'train_err': [], 'test_err': [], 'sparsity': []}
results_L1_s = {'train_err': [], 'test_err': [], 'sparsity': []}

for lam in lambdas_ex3:
    # Ridge
    w_L2 = ridge_regression(X_train, y_train, lam=lam)
    results_L2_s['train_err'].append(mse(y_train, predict(X_train, w_L2)))
    results_L2_s['test_err'].append(mse(y_test, predict(X_test, w_L2)))
    results_L2_s['sparsity'].append(np.sum(w_L2 == 0))
    
    # Lasso
    w_L1 = lasso_coordinate_descent(X_train, y_train, lam=lam)
    results_L1_s['train_err'].append(mse(y_train, predict(X_train, w_L1)))
    results_L1_s['test_err'].append(mse(y_test, predict(X_test, w_L1)))
    results_L1_s['sparsity'].append(np.sum(w_L1 == 0))

print("λ | Ridge Test | Lasso Test | Ridge Zeros | Lasso Zeros")
for i, lam in enumerate(lambdas_ex3):
    print(f"{lam:.3f} | "
          f"{results_L2_s['test_err'][i]:.4f} | "
          f"{results_L1_s['test_err'][i]:.4f} | "
          f"{results_L2_s['sparsity'][i]} | "
          f"{results_L1_s['sparsity'][i]}")
    

# Concrete
""""
print("Concrete Lasso Lambda Interval Test")
for lam in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    w_L1 = lasso_coordinate_descent(X_train_conc_norm, y_train_conc_norm, lam=lam)
    zeros = np.sum(w_L1 == 0)
    test_err = mse(y_test_conc_norm, predict(X_test_conc_norm, w_L1))
    print(f"λ={lam} | Zero weights: {zeros}/8 | Test MSE: {test_err:.4f}")
"""

results_L2_c = {'train_err': [], 'test_err': [], 'sparsity': []}
results_L1_c = {'train_err': [], 'test_err': [], 'sparsity': []}

for lam in lambdas_ex3:
    # Ridge
    w_L2 = ridge_regression(X_train_conc_norm, y_train_conc_norm, lam=lam)
    results_L2_c['train_err'].append(mse(y_train_conc_norm, predict(X_train_conc_norm, w_L2)))
    results_L2_c['test_err'].append(mse(y_test_conc_norm, predict(X_test_conc_norm, w_L2)))
    results_L2_c['sparsity'].append(np.sum(w_L2 == 0))
    
    # Lasso
    w_L1 = lasso_coordinate_descent(X_train_conc_norm, y_train_conc_norm, lam=lam)
    results_L1_c['train_err'].append(mse(y_train_conc_norm, predict(X_train_conc_norm, w_L1)))
    results_L1_c['test_err'].append(mse(y_test_conc_norm, predict(X_test_conc_norm, w_L1)))
    results_L1_c['sparsity'].append(np.sum(w_L1 == 0))

print("λ | Ridge Test | Lasso Test | Ridge Zeros | Lasso Zeros")
for i, lam in enumerate(lambdas_ex3):
    print(f"{lam:.3f} | "
          f"{results_L2_c['test_err'][i]:.4f} | "
          f"{results_L1_c['test_err'][i]:.4f} | "
          f"{results_L2_c['sparsity'][i]} | "
          f"{results_L1_c['sparsity'][i]}")
    

# Graphs
# Synthetic L1 vs L2 Graph
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('(Synthetic) L1 vs L2', fontsize=13)

axes[0].plot(lambdas_ex3, results_L2_s['test_err'],
             marker='o', label='Ridge (L2)', color='pink')
axes[0].plot(lambdas_ex3, results_L1_s['test_err'],
             marker='o', label='Lasso (L1)', color='green')
axes[0].set_xlabel('λ')
axes[0].set_ylabel('Test MSE')
axes[0].set_title('Test Error')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(lambdas_ex3, results_L1_s['sparsity'],
             marker='o', label='Lasso (L1)', color='green')
axes[1].plot(lambdas_ex3, results_L2_s['sparsity'],
             marker='o', label='Ridge (L2)', color='pink')
axes[1].set_xlabel('λ')
axes[1].set_ylabel('# of Zero Weights')
axes[1].set_title('Sparsity')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Concrete L1 vs L2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('(Concrete) L1 vs L2', fontsize=13)

axes[0].plot(lambdas_ex3, results_L2_c['test_err'],
             marker='o', label='Ridge (L2)', color='pink')
axes[0].plot(lambdas_ex3, results_L1_c['test_err'],
             marker='o', label='Lasso (L1)', color='green')
axes[0].set_xlabel('λ')
axes[0].set_ylabel('Test MSE')
axes[0].set_title('Test Error')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(lambdas_ex3, results_L1_c['sparsity'],
             marker='o', label='Lasso (L1)', color='green')
axes[1].plot(lambdas_ex3, results_L2_c['sparsity'],
             marker='o', label='Ridge (L2)', color='pink')
axes[1].set_xlabel('λ')
axes[1].set_ylabel('# of Zero Weights')
axes[1].set_title('Sparsity')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()