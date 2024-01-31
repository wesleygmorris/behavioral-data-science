library(ggplot2)
library(caret)

df <- read.csv('C:\\Users\\morriwg1\\OneDrive - Vanderbilt\\Documents\\GitHub\\behavioral-data-science\\behavioral-data-science\\data\\student_math\\student_math_clean.csv')
colnames(df)

set.seed(42)

training <- sample(
    1:nrow(df),
    round(nrow(df) * 0.8),
    replace=FALSE
)

data_train <- df[training,]
data_test <- df[-training,]

# Regression

fit <- lm(final_grade ~ study_time + class_failures + school_support + 
    family_support + higher_ed + internet_access + health + absences, 
    data=data_train)

plot(fit)
summary(fit)

predict_test <- predict(fit, newdata=data_test)

sqrt(mean((predict_test - data_test$final_grade)^2))

ggplot(data=df, aes(y=final_grade, x=higher_ed)) +
    geom_bar(stat='summary', fun='mean') +
    stat_summary(fun.data = mean_se, geom = "errorbar")

data <- data.frame(predict = predict_test, actual = data_test$final_grade)


ggplot(data=data, aes(x=actual, y=predict)) +
    geom_point() +
    geom_smooth(method='lm', formula= y~x) +
    labs(title='Predicted vs Actual Data')

## Bootstrapping

set.seed(42)
library(caret)
library(ggplot2)

colnames(df)

training <- sample(
    1:nrow(df),
    round(nrow(df) * 0.8),
    replace=FALSE
)

data_train <- df[training,]
data_test <- df[-training,]

train_control <- trainControl(
    method = 'boot',
    number = 100
)

data_boot <- train(
    form = final_grade ~ study_time + class_failures + school_support + 
        family_support + higher_ed + internet_access + health + absences,
    data = data_train,
    method = 'lm',
    metric = 'RMSE',
    trControl = train_control
)
data_boot


predict_test <- as.vector(predict(data_boot, data_test))

sqrt(mean((predict_test-data_test$final_grade)^2))

data <- data.frame(predict = predict_test, actual = data_test$final_grade)

ggplot(data=data, aes(x=actual, y=predict)) +
    geom_point() +
    geom_smooth(method='lm', formula= y~x) +
    labs(title='Predicted vs Actual Data')

varImp(data_boot)$importance



# K-folds
library(caret)
set.seed(42)

train_control <- trainControl(
    method = 'cv',
    number = 5
)

data_cv <- train(
    form = final_grade ~ study_time + class_failures + school_support + 
        family_support + higher_ed + internet_access + health + absences,
    data = data_train,
    method = 'lm',
    metric = 'RMSE',
    trControl = train_control
)

data_cv

predict_test <- as.vector(predict(data_cv, data_test))

sqrt(mean((predict_test-data_test$final_grade)^2))

data <- data.frame(predict = predict_test, actual = data_test$final_grade)

ggplot(data=data, aes(x=actual, y=predict)) +
    geom_point() +
    geom_smooth(method='lm', formula= y~x) +
    labs(title='Predicted vs Actual Data')

varImp(data_boot)$importance


###########################
### Logistic Regression ###
###########################

data_train$extra_paid_classes <- as.factor(data_train$extra_paid_classes)
data_test$extra_paid_classes <- as.factor(data_test$extra_paid_classes)

## Normal logistic regression
fit <- glm(extra_paid_classes ~ study_time + class_failures + school_support + 
    family_support + higher_ed + internet_access + health + absences, 
    data=data_train, family=binomial)

plot(fit)
summary(fit)

### get probabilities
predict_props <- predict(fit, newdata=data_test)
### convert probabilities into predictions
predict_test <- ifelse(predict_props >= 0.5, 'yes', 'no')
predict_test <- as.factor(predict_test)
### check it out
confusionMatrix(predict_test, data_test$extra_paid_classes)


## Bootstrapping

set.seed(42)
library(caret)
library(ggplot2)

### get the sample
training <- sample(
    1:nrow(df),
    round(nrow(df) * 0.8),
    replace=FALSE
)
### make training and tes
data_train <- df[training,]
data_test <- df[-training,]

### define train control
train_control <- trainControl(
    method = 'boot',
    number = 100
)

### train it
data_boot <- train(
    form = extra_paid_classes ~ study_time + class_failures + school_support + 
        family_support + higher_ed + internet_access + health + absences,
    data = data_train,
    method = 'glm',
    family = 'binomial',
    metric = 'Accuracy',
    trControl = train_control
)
data_boot

### make the predictions
predict_test <- predict(data_boot, newdata=data_test)
as.factor(data_test$extra_paid_classes)
data_test$extra_paid_classes

### check it out
confusionMatrix(predict_test, as.factor(data_test$extra_paid_classes))
varImp(data_boot)$importance



## K-folds
library(caret)
set.seed(42)
### define the train control
train_control <- trainControl(
    method = 'cv',
    number = 5
)
### train it
data_cv <- train(
    form = extra_paid_classes ~ study_time + class_failures + school_support + 
        family_support + higher_ed + internet_access + health + absences,
    data = data_train,
    method = 'glm',
    family = 'binomial',
    metric = 'Accuracy',
    trControl = train_control
)

### make the predictions
predict_test <- predict(data_cv, newdata=data_test)
as.factor(data_test$extra_paid_classes)
data_test$extra_paid_classes

### check it out
confusionMatrix(predict_test, as.factor(data_test$extra_paid_classes))
varImp(data_cv)$importance