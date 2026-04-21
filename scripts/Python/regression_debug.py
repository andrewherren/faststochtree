# Load libraries
import time
import numpy as np
import faststochtree as fst
import matplotlib.pyplot as plt

# Generate data
rng = np.random.default_rng()
n = 50000
n_test = 500
p = 50
X = rng.uniform(low=0., high=1., size=(n,p))
X_test = rng.uniform(low=0., high=1., size=(n_test,p))
y = np.sin(X[:,0]) + rng.standard_normal(n)
f_test = np.sin(X_test[:,0])
y_test = f_test + rng.standard_normal(n_test)

# Fit BART / XBART
start = time.time()
# model = fst.fit_bart(X, y, X_test)
config = fst.BARTConfig()
config.num_threads = 8
model = fst.fit_xbart(X, y, X_test, config=config)
end = time.time()
print(end - start)

# Predict on test set
yhat = np.mean(model.predict(X_new = X_test), axis=0)
lo, hi = min(yhat.min(), f_test.min()), max(yhat.max(), f_test.max())
plt.scatter(yhat, f_test)
plt.plot([lo, hi], [lo, hi], color="red", linestyle="dashed", linewidth=2)
