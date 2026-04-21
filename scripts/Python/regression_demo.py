# Load libraries
import time
import math
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
f = np.sin(X[:,0])
y = f + rng.standard_normal(n)
f_test = np.sin(X_test[:,0])
y_test = f_test + rng.standard_normal(n_test)

# Fit BART / XBART
start = time.time()
num_trees = 200
# mcmc_config = fst.BARTConfig()
# mcmc_config.num_trees = num_trees
# mcmc_config.leaf_prior_var = 1.0 / num_trees # Calibrate leaf prior scale to 1 / num_trees
# model = fst.fit_bart(X, y, X_test, config=mcmc_config)
gfr_config = fst.BARTConfig()
gfr_config.num_threads = 8 # Replace with however many threads you want to use
gfr_config.p_eval = int(math.sqrt(p)) # GFR default: evaluate sqrt(p) features per node
gfr_config.num_trees = num_trees
gfr_config.leaf_prior_var = 1.0 / num_trees # Calibrate leaf prior scale to 1 / num_trees
model = fst.fit_xbart(X, y, X_test, config=gfr_config)
end = time.time()
print(end - start)

# Extract test set predictions
yhat_test = np.mean(model.test_samples, axis=0)
lo, hi = min(yhat_test.min(), f_test.min()), max(yhat_test.max(), f_test.max())
plt.scatter(yhat_test, f_test)
plt.plot([lo, hi], [lo, hi], color="red", linestyle="dashed", linewidth=2)

# Alternatively, can obtain new test predictions from the model
yhat_new = np.mean(model.predict(X_new = X_test), axis=0)
lo, hi = min(yhat_test.min(), yhat_new.min()), max(yhat_test.max(), yhat_new.max())
plt.clf()
plt.scatter(yhat_new, yhat_test)
plt.plot([lo, hi], [lo, hi], color="red", linestyle="dashed", linewidth=2)

# Inspect global error variance samples
variance_samples = model.sigma2_samples
plt.clf()
plt.plot(variance_samples)
