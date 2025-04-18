# Question 2
library(mice)

# Load data
load("E:/dataex2.Rdata")

# Number of imputations and iterations
M <- 20
num_iter <- 100

# Initialize confidence interval matrices
CI_stochastic <- CI_bootstrap <- matrix(NA, nrow = num_iter, ncol = 2)

# Function to perform multiple imputation and extract confidence intervals
impute_and_extract_CI <- function(data, method) {
  imp <- mice(data, m = M, seed = 1, method = method, printFlag = FALSE)
  fit <- with(imp, lm(y ~ x))
  pool_summary <- summary(pool(fit), conf.int = TRUE)
  return(as.numeric(pool_summary[2, c("2.5 %", "97.5 %")]))
}

# Run imputations
for (i in 1:num_iter) {
  data_i <- dataex2[, , i]
  colnames(data_i) <- c("x", "y")
  CI_stochastic[i, ] <- impute_and_extract_CI(data_i, "norm")
  CI_bootstrap[i, ] <- impute_and_extract_CI(data_i, "norm.boot")
}

# Compute empirical coverage probabilities
coverage_rate <- function(CI) mean(CI[, 1] <= 3 & CI[, 2] >= 3)

cat("Empirical Coverage Probability (Stochastic Regression Imputation):", coverage_rate(CI_stochastic), "\n")
cat("Empirical Coverage Probability (Bootstrap-Based Imputation):", coverage_rate(CI_bootstrap), "\n")


#Question 3
install.packages("maxLik")
library(maxLik)
load("E:/dataex3.Rdata")

# Define the log-likelihood function
log_likelihood <- function(mu) {
  x<- dataex3[, 1]
  r<- dataex3[, 2]
  sigma_sq <- 1.5^2
  pdf <- dnorm(x, mean = mu, sd = sqrt(sigma_sq))
  cdf <- pnorm(x, mean = mu, sd = sqrt(sigma_sq))
  sum(r * log(pdf) + (1 - r) * log(cdf))
}

# Initial guess for mu
initial_mu <- mean(x)

# Maximize the log-likelihood using maxLik
result <- maxLik(
  logLik = log_likelihood,
  start = c(mu = initial_mu)
)
summary(result)
# MLE of mu
mle_mu <- result$estimate

# Output the result
cat("Maximum Likelihood Estimate of mu:", mle_mu, "\n") 


#Question 5

# Load the dataset
load("E:/dataex5.Rdata")
set.seed(20)

# Separate fully observed data and missing data
observed_data <- na.omit(dataex5)  # Fully observed data
missing_data <- dataex5[is.na(dataex5$Y), ]  # Data with missing values

# E-step function: Impute missing values
impute_missing_values <- function(missing_data, beta) {
  # Compute the expected value for missing Y
  missing_data$Y <- exp(beta[1] + missing_data$X * beta[2]) / 
    (1 + exp(beta[1] + missing_data$X * beta[2]))
  return(missing_data)
}

# EM Algorithm Initialization
beta_current <- c(beta0 = rnorm(1), beta1 = rnorm(1))  # Initial parameter values
max_iterations <- 100  # Maximum number of iterations
tolerance <- 1e-6  # Convergence threshold
converged <- FALSE  # Convergence flag
iteration_count <- 0  # Iteration counter

while (!converged && iteration_count < max_iterations) {
  # E-step: Impute missing values
  filled_data <- impute_missing_values(missing_data, beta_current)
  complete_data <- rbind(observed_data, filled_data)  # Combine observed and imputed data
  
  # M-step: Maximize the log-likelihood function
  log_likelihood_function <- function(params) {
    beta0 <- params[1]
    beta1 <- params[2]
    
    x_values <- complete_data$X
    y_values <- complete_data$Y
    
    # Compute probability
    predicted_prob <- exp(beta0 + x_values * beta1) / 
      (1 + exp(beta0 + x_values * beta1))
    
    # Compute log-likelihood
    log_likelihood <- sum(y_values * log(predicted_prob) + 
                            (1 - y_values) * log(1 - predicted_prob))
    return(log_likelihood)
  }
  
  # Use maxLik to maximize the log-likelihood function
  optimization_result <- maxLik(
    logLik = log_likelihood_function,
    start = beta_current,
    method = "NR"  # Newton-Raphson method
  )
  
  # Update parameters
  beta_new <- coef(optimization_result)
  
  # Check convergence
  if (max(abs(beta_new - beta_current)) < tolerance) {
    converged <- TRUE
  }
  
  # Update current parameters
  beta_current <- beta_new
  iteration_count <- iteration_count + 1
}
summary(result)
# Output the results
cat("Maximum Likelihood Estimate of beta:\n")
cat("beta0:", beta_current[1], "\n")
cat("beta1:", beta_current[2], "\n")
cat("Number of iterations:", iteration_count, "\n")
