# Load library
library(faststochtree)

# Generate data
n <- 50000
n_test <- 500
p <- 50
X <- matrix(runif(n * p), ncol = p)
X_test <- matrix(runif(n_test * p), ncol = p)
y <- sin(X[, 1]) + rnorm(n)
f_test <- sin(X_test[, 1])
y_test <- f_test + rnorm(n_test)

# Fit BART / XBART
start <- Sys.time()
num_trees <- 200
# mcmc_config <- bart_config(
#   num_trees = num_trees,
#   leaf_prior_var = 1 / num_trees
# )
# model <- faststochtree::fit_bart(X, y, X_test, config = mcmc_config)
gfr_config <- bart_config(
  num_threads = 8L, 
  p_eval = as.integer(sqrt(p)), 
  num_trees = num_trees,
  leaf_prior_var = 1 / num_trees
)
model <- faststochtree::fit_xbart(X, y, X_test, config = gfr_config)
end <- Sys.time()
print(end - start)

# Predict on test set
yhat <- predict(model, newdata = X_test)
plot(colMeans(yhat), f_test)
abline(0, 1)
