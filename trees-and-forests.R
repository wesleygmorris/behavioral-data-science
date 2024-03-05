#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### PSY-GS 8875 | Week 6: Trees and Forests ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

# Load packages
library(caret); library(ggplot2)
library(ranger); library(ggpubr)
library(rpart); library(rpart.plot)

# Load data
schizotypy <- read.csv(".\\data\\schizotypy\\share_430n_interview.csv")

# Wisconsin Schizotypy Scales
# py = physical anhedonia
# sa = social anhdenoia
# pb = perception aberration
# mi = magical ideation

# Outcomes of interest
# gas = global functioning (0-100; higher = better)
# psx_high = psychotic-like experiences (positive symptoms)
# nsm_tot = negative symptoms

# Fill in `NA` with 0
schizotypy[,4:63][is.na(schizotypy[,4:63])] <- 0

hist(schizotypy$gas)

#%%%%%%%%%%%%%%%%%
## Regression ----
#%%%%%%%%%%%%%%%%%

colnames(schizotypy)

# Linear regression
hist(schizotypy$gas)
gas_lm <- lm(
  gas ~ ., data = schizotypy[,c(4:64)]
); summary(gas_lm)

# Plot linear model

# Perform tree
tree <- rpart(
  gas ~ ., data = schizotypy[,c(4:66)],
  parms = list(split = "RMSE")
)

# Identify best cp value
tree$cptable[
  which.min(tree$cptable[,"rel error"]),
  "CP" # find lowest error cp
]

# Prune tree
prune_tree <- prune(tree, cp = 0.01)

# Plot
prp(prune_tree, extra = 1)

# Using rpart.plot
rpart.plot(
  prune_tree,
  box.palette = "GnBu", # color scheme
  branch.lty = 3, # dotted branch lines
  shadow.col = "gray", # shadows under the node boxes
  nn = TRUE
)

# Prediction
tree_predictions <- predict(prune_tree)
tree_predictions

cor.test(tree_predictions, schizotypy$gas)
plot(tree_predictions, schizotypy$gas)

continuous_accuracy <- function(prediction, observed) { 
  # Compute square error 
  square_error <- (prediction - observed)^2 
  # Return metrics 
  return( c( R2 = 1 - ( sum(square_error, na.rm = TRUE) / sum((observed - mean(observed, na.rm = TRUE))^2, na.rm = TRUE) ), 
             RMSE = sqrt(mean(square_error, na.rm = TRUE)) ) ) } 

# Usage (replace with appropriate values) 
continuous_accuracy(tree_predictions, schizotypy$gas)


# Set seed for reproducibility
set.seed(42)

# Random forest cross-validation for parameters
tictoc::tic()
store_caret <- train(
  x = schizotypy[,4:63],
  y = schizotypy$gas,
  method = "ranger",
  metric = "RMSE", # better for imbalanced datasets
  trControl = trainControl(
    method = "cv", number = 5
  ),
  tuneGrid = expand.grid(
    mtry = seq(1, 60, 1), # 1:ncol(data)
    min.node.size = seq(1, 10, 1), # 1-10 is usually good
    splitrule = "variance" # classification
  ),
  num.trees = 500, # keep at 500 for the initial search
  importance = "impurity" # set up for later
); store_caret
tictoc::toc()
# This process will take a long time
# On my laptop (16 cores): 83.822 sec elapsed

# With the `mtry` and `min.node.size` parameters,
# search over `num.trees`
trees <- c(10, 50, 100, 250, 500, 1000, 1500)

# Store results
results <- vector("list", length(trees))

# Perform cross-validation (will be much faster than before)
for(i in seq_along(trees)){

  # Perform
  results[[i]] <- train(
    x = schizotypy[,4:64],
    y = schizotypy$gas,
    method = "ranger",
    metric = "RMSE", # better for unbalanced datasets
    trControl = trainControl(
      method = "cv", number = 5
      #, sampling = "smote"
      # For class imbalances:
      # https://topepo.github.io/caret/subsampling-for-class-imbalances.html
    ),
    tuneGrid = data.frame(
      mtry = 43, splitrule = "variance",
      min.node.size = 9
    ),
    num.trees = trees[i], # search over trees
    importance = "impurity" # set up for later
  )$results

}

# Combine results
combined <- do.call(rbind.data.frame, results)
combined$num.trees <- trees
combined

# 250 trees has the best accuracy/kappa

# Get final model
ranger_model <- ranger(
  formula = gas ~ .,
  data = schizotypy[,c(4:64)],
  mtry = 43, splitrule = "variance", # 43, splitrule = "gini",
  min.node.size = 9, num.trees = 250,
  importance = "impurity", seed = 42 # for reproducibility
)

# Get predictions
ranger_predictions <- predict(ranger_model, data = schizotypy)$predictions
ranger_predictions

# Plot predictions
cor.test(ranger_predictions, schizotypy$gas)
plot(ranger_predictions, schizotypy$gas)
continuous_accuracy(ranger_predictions, schizotypy$gas)
# Check out importance
ranger_imp <- importance(ranger_model); ranger_imp

round(ranger_imp, 2)

# Visualize importance
ggdotchart(
  data = data.frame(
    Importance = round(ranger_imp, 2),
    Variable = names(ranger_imp),
    Dimension = rep(
      c(
        "Physical Anhedonia", "Perceptual Aberration",
        "Magical Ideation", "Social Anhedonia"
      ), each = 15
    )
  ),
  x = "Variable", y = "Importance", color = "Dimension",
  dot.size = 5, add = "segments", label = "Importance",
  group = "Dimension", # for within-dimension comparison
  font.label = list(size = 8, vjust = 0.5, color = "black", face = "bold")
) +
  scale_y_continuous(limits = c(0, 2500), n.breaks = 8, expand = c(0, 0)) +
  guides(color = guide_legend(title.position = "left", nrow = 2)) +
  theme(
    legend.position = "top",
    legend.title = element_text(size = 14, hjust = 0.5),
    legend.text = element_text(size = 10),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  ) +
  coord_flip()

# Look at first tree
treeInfo(ranger_model, tree = 1)

# For extra interpretability
library(randomForestExplainer)

# Explain forest
explain_forest(
  ranger_model, data = schizotypy,
  path = paste0(getwd(), "/schizotypy_negative")
)

library(pandoc)
pandoc_available()
rmarkdown::pandoc_available
install.packages('pandoc')


# Predict gas from all schizotypy items
# Task is regression and differs from the in-class example and activity
# perform linear regression and evaluation with RMSE and R2
# Perform random forest regression (use cross-validation)
# and grid search to select hyperparameters) and evaluate with RMSE and R2
# Compare models on:
## Significant (linear regression) and important(random forest) variables
## RMSE and R2 (which is better?)

