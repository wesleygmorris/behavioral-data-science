#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### PSY-GS 8875 | Week 2: Regression ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


# Load packages
library(ggplot2); library(latentFactoR)
source("C:\\Users\\morriwg1\\OneDrive - Vanderbilt\\Documents\\GitHub\\behavioral-data-science\\behavioral-data-science\\source\\glm_gradient_descent.R")

# Set seed to ensure reproducible results
set.seed(42)

#%%%%%%%%%%%%%%%%
## Your turn ----
#%%%%%%%%%%%%%%%%

# Your goal is to "fine-tune" the gradient descent algorithm
# to get the beta coefficients to match the coefficients
# determine by OLS

# Here's your data
# Generate data
X <- rnorm(1000, mean = 0, sd = 0.10)
Y <- X + rnorm(1000)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# First we will define the function that will work
# to find the line of best fit using gradient descent
# for both logistic and linear regression.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Define the function. This can be used either for linear or sigmoid activation.
bootleg_gridsearch <- function(activation, max_num_iters=1000){
  # Set the grid of values that we will search over.
  learning_rate_range = seq(0, 1, by=0.1)
  max_iter_range = seq(100, max_num_iters, by=100)
  total=length(learning_rate_range)*length(max_iter_range)*length(activation_range)
  
  # Initialize the vectors
  learning_rates <- vector()
  max_iters <- vector()
  costs <- vector()
  
  # Initialize the progress bar
  pb = txtProgressBar(min = 0, max = total, initial = 0) 
  counter = 1
  total = length(learning_rate_range)*length(max_iter_range)*length(activation_range)
  
  # Main hyperparameter search loop
  for (lr in learning_rate_range){
    for (mi in max_iter_range){
      setTxtProgressBar(pb,counter)
      best_result <- glm_gradient_descent(X, Y, learning_rate=lr, max_iter=mi, activation='linear')$best_result
      cost <- best_result[2]
      costs[counter] <- cost
      learning_rates[counter] <- lr
      max_iters[counter] <- mi
      counter <- counter + 1
    }
  }  
  hyperparameter_df <- data.frame(costs=costs, learning_rates=learning_rates, max_iters=max_iters)
  best_one <- hyperparameter_df[which.min(abs(hyperparameter_df$cost)),]

  ## Do gradient descent using the best hyperparameters
  gd <- glm_gradient_descent(
    X, Y, 
    learning_rate=best_one$learning_rates, 
    max_iter=best_one$max_iters, 
    activation=activation)
  
  return(gd)
}

#%%%%%%%%%%%%%%%%%%%%%%%
# Linear Regression ----
#%%%%%%%%%%%%%%%%%%%%%%%
# glm_gradient_descent(
#   X, Y, learning_rate = 0.2,
#   max_iter = 7800, activation = "linear"
# )

# Generate data
X <- rnorm(1000, mean = 0, sd = 0.10)
Y <- X + rnorm(1000)

results <- bootleg_gridsearch('linear', 1000)
print(results$best_result)
print(coef(lm(Y ~ X)))

## Plot the gradient descent
results_frame <- data.frame(results$results)
ggplot(results_frame, aes(x=iteration, y=cost)) +
  geom_line()



#%%%%%%%%%%%%%%%%%%%%%%%%%%
## Logistic regression ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%
# Convert Y to binary
Y <- categorize(Y, 2) - 1
# Adjust the parameters of the gradient descent
# to get near the logistic regression output

results <- bootleg_gridsearch('sigmoid', 5000)
print(results$best_result)
print(coef(glm(Y ~ X, family='binomial')))

## Plot the gradient descent
results_frame <- data.frame(results$results)
ggplot(results_frame, aes(x=iteration, y=cost)) +
  geom_line()

