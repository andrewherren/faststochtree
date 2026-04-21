# Load library
library(faststochtree)

# Generate data
n <- 50000
n_test <- 500
p <- 50
X <- matrix(runif(n * p), ncol = p)
X_test <- matrix(runif(n_test * p), ncol = p)
f <- sin(X[, 1])
y <- f + rnorm(n)
f_test <- sin(X_test[, 1])
y_test <- f_test + rnorm(n_test)

# Fit BART / XBART
start <- Sys.time()
num_trees <- 200
# mcmc_config <- bart_config(
#   num_trees = num_trees,
#   leaf_prior_var = 1 / num_trees # Calibrate leaf prior scale to 1 / num_trees
# )
# model <- faststochtree::fit_bart(X, y, X_test, config = mcmc_config)
gfr_config <- bart_config(
  num_threads = 8L, # Replace with however many threads you want to use
  p_eval = as.integer(sqrt(p)), # GFR default: evaluate sqrt(p) features per node
  num_trees = num_trees, 
  leaf_prior_var = 1 / num_trees # Calibrate leaf prior scale to 1 / num_trees
)
model <- faststochtree::fit_xbart(X, y, X_test, config = gfr_config)
end <- Sys.time()
print(end - start)

# Extract test set predictions
yhat_test <- colMeans(test_samples(model))
plot(yhat_test, f_test)
abline(0, 1)

# Alternatively, can obtain new test predictions from the model
yhat_new <- colMeans(predict(model, newdata = X_test))
plot(yhat_test, yhat_new)
abline(0, 1)

# Inspect global error variance samples
variance_samples <- sigma2_samples(model)
plot(variance_samples)
