---
title: "Practical Machine Learning Course Project"
author: "Marieke Reus"
date: "11-3-2021"
output: 
  html_document: 
    keep_md: yes
---

 
This project analyzes data from accelerometers. The goal of the project is to predict the manner in which subjects did an exercise (the "classe" variable) with a prediction model.


## Data preprocessing
First, the required packages were loaded and the training and test data was read into R.

```r
# Load packages
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
# Read data into R
training_data <- read.csv("pml-training.csv")
test_data <- read.csv("pml-testing.csv")
```

The seed was set to account for reproducibility. The training data was split into a training and test set.

```r
# Split the training data into a training and test set
set.seed(123)
inTrain <- createDataPartition(training_data$classe, p=0.6, list=FALSE)
train <- training_data[inTrain, ]
test <- training_data[-inTrain, ]
```

The unnecessary variables from the datasets were removed. Those are variables with nearly zero variance, variables that are mostly NA, and variables used for ID purposes.

```r
# Remove variables with nearly zero variance
NZV <- nearZeroVar(train)
train <- train[, -NZV]
test <- test[, -NZV]

# Remove variables that are mostly NA
NAvar <- colMeans(is.na(train)) > 0.95
train <- train[, NAvar==FALSE]
test <- test[, NAvar==FALSE]

# Remove variables that are used for ID purposes and do not make sense to use for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp)
train <- train[, -(1:5)]
test <- test[, -(1:5)]
```


## Prediction Models
Two different prediction models (decision tree and random forest) were fitted on the training set and their performance on the test set was evaluated with a confusion matrix.

### Decision Tree
A decision tree model was fitted on the training set, with 3-fold cross-validation.

```r
# Fit a decision tree model on the training set
controlDT <- trainControl(method="cv", number=3, verboseIter=FALSE)
fitDT <- train(classe ~ ., data=train, method="rpart", trControl=controlDT)
fitDT$finalModel
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 10809 7472 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -26.65 1077   52 A (0.95 0.048 0 0 0) *
##      5) pitch_forearm>=-26.65 9732 7420 A (0.24 0.23 0.21 0.2 0.12)  
##       10) num_window>=45.5 9292 6980 A (0.25 0.24 0.22 0.2 0.092)  
##         20) num_window< 241.5 2126 1015 A (0.52 0.13 0.1 0.2 0.055) *
##         21) num_window>=241.5 7166 5207 B (0.17 0.27 0.26 0.2 0.1)  
##           42) magnet_dumbbell_z< -25.5 1958 1048 A (0.46 0.35 0.047 0.13 0.015)  
##             84) num_window< 686.5 949  193 A (0.8 0.16 0.0032 0.032 0.0084) *
##             85) num_window>=686.5 1009  485 B (0.15 0.52 0.089 0.22 0.022) *
##           43) magnet_dumbbell_z>=-25.5 5208 3462 C (0.056 0.25 0.34 0.23 0.14)  
##             86) magnet_dumbbell_x< -447.5 3704 2055 C (0.063 0.16 0.45 0.24 0.087) *
##             87) magnet_dumbbell_x>=-447.5 1504  818 B (0.038 0.46 0.064 0.18 0.26) *
##       11) num_window< 45.5 440   90 E (0 0 0 0.2 0.8) *
##    3) roll_belt>=130.5 967   11 E (0.011 0 0 0 0.99) *
```

The decision tree model was used to predict the classe variable of the test set and the performance of the model was evaluated with the confusion matrix.

```r
# Predict classe in the test set
predictDT <- predict(fitDT, newdata=test)

# Calculate the confusion matrix
confusionMatrix(predictDT, factor(test$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1891  287  170  288   79
##          B  163  856  131  310  258
##          C  175  375 1067  632  207
##          D    0    0    0    0    0
##          E    3    0    0   56  898
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6006          
##                  95% CI : (0.5896, 0.6114)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4893          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8472   0.5639   0.7800   0.0000   0.6227
## Specificity            0.8532   0.8638   0.7856   1.0000   0.9908
## Pos Pred Value         0.6965   0.4983   0.4344      NaN   0.9383
## Neg Pred Value         0.9335   0.8920   0.9442   0.8361   0.9210
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2410   0.1091   0.1360   0.0000   0.1145
## Detection Prevalence   0.3460   0.2190   0.3130   0.0000   0.1220
## Balanced Accuracy      0.8502   0.7138   0.7828   0.5000   0.8068
```
The accuracy of the decision tree model is 0.6006, so the expected out of sample error is 0.3994.

### Random Forest
A random forest model was fitted on the training set, with 3-fold cross-validation.

```r
# Fit a random forest model on the training set
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
fitRF <- train(classe ~ ., data=train, method="rf", trControl=controlRF)
fitRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.25%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3348    0    0    0    0 0.000000000
## B    7 2272    0    0    0 0.003071523
## C    0    5 2049    0    0 0.002434275
## D    0    0   14 1916    0 0.007253886
## E    0    0    0    3 2162 0.001385681
```

The random forest model was used to predict the classe variable of the test set and the performance of the model was evaluated with the confusion matrix.

```r
# Predict classe in the test set
predictRF <- predict(fitRF, newdata=test)

# Calculate the confusion matrix
confusionMatrix(predictRF, factor(test$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    2    0    0    0
##          B    0 1514    6    0    0
##          C    0    2 1362    2    0
##          D    0    0    0 1283    3
##          E    1    0    0    1 1439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9965, 0.9987)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9973          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9974   0.9956   0.9977   0.9979
## Specificity            0.9996   0.9991   0.9994   0.9995   0.9997
## Pos Pred Value         0.9991   0.9961   0.9971   0.9977   0.9986
## Neg Pred Value         0.9998   0.9994   0.9991   0.9995   0.9995
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1930   0.1736   0.1635   0.1834
## Detection Prevalence   0.2846   0.1937   0.1741   0.1639   0.1837
## Balanced Accuracy      0.9996   0.9982   0.9975   0.9986   0.9988
```
The accuracy of the random forest model is 0.9975, so the expected out of sample error is 0.0025.

The best prediction model is thus the random forest model, with an accuracy of 0.9975 and an expected out of sample error of 0.0025.


## Predictions
The random forest model was used to predict the classe variable for the 20 cases in the test data.

```r
# Predict classe in the test data
pred <- predict(fitRF, test_data)
print(pred)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
