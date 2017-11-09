Prediction Assignment
================
Vitawat Ngammuangpak
10/27/2017

1. Background
=============

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

2. Objective of project
=======================

The object want to predict the class of preformance by use data from accelerometers on belt, forearm, arm and dumbell of 6 participants that they were asked to perform barbell lifts correctly and incorrectly in 5 different ways.(predict 20 value on test set )

3. Work process
===============

-   Step 1 Data preparing: import and clean data
-   Step 2 Fit model: rpart(), randomForest() and svm()
-   Step 3 Model selection: compare accuray and select the highest accuracy.
-   Step 4 Prediction

4. Data preparing
=================

**4.1 Import data**

Found 19,622 obs. in pml\_training and 20 obs. in pml\_testing. Both data set have 160 varaibles, found some varaible have "\#DIV/0!" and a lot of missing.

    ## [1] "Importing training data.........."

    ## Warning: Missing column names filled in: 'X1' [1]

    ## Warning: 185 parsing failures.
    ##  row               col expected  actual
    ## 2231 kurtosis_roll_arm a double #DIV/0!
    ## 2231 skewness_roll_arm a double #DIV/0!
    ## 2255 kurtosis_roll_arm a double #DIV/0!
    ## 2255 skewness_roll_arm a double #DIV/0!
    ## 2282 kurtosis_roll_arm a double #DIV/0!
    ## .... ................. ........ .......
    ## See problems(...) for more details.

    ## [1] 19622   160

    ## [1] "Importing testing data.........."

    ## Warning: Missing column names filled in: 'X1' [1]

    ## [1]  20 160

**4.2 Data cleaning**

First of all, we merge 2 data set together, then remove unuse varaible such as X1, user\_name, raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, cvtd\_timestamp, new\_window and num\_window.

For some data problem, we manage as follows

-   Change value "\#DIV/0!" to NA
-   Delete varaible where contain all NA
-   Delete varaible where contain all zero
-   For varaible contain a lot of NA, we decide to remove varaible that have NA more than 50% of total case.
-   For missing value, replace missing value with mean of that varaible

After clean the data, we have new\_pml\_training and new\_pml\_testing ready to analysis in the next step.

``` r
# Change value "#DIV/0!" to NA
total[total =="#DIV/0!"] <- NA 

# Remove varaible that contain all NA (NA column)
total <- total[ ,colSums(is.na(total))<nrow(total)] 

# Remove varible that contain all zero (zero value column)
total <- total[ ,colSums(total != 0, na.rm = TRUE) > 0]  

# Remove varible which contain NA more than 50 %
total <- total[ ,colSums(is.na(total))/nrow(total) < 0.5]
```

5. Fit Model
============

We used randonForest(), rpart() and svm() to find the prediction model. For each method, steps are as follow

-   Devided training data to trainData 70% and testData 30%, use createDataPartition() from caret package by defined p = 0.70
-   Use trainData to find model (in function, use "classe" is dependent varaible, the rest varaible is predictors and use defualt value for condition)
-   Use model to predict "classe" on testData.
-   Use confusionMatrix() to find accuracy value of model.

``` r
library(caret)
set.seed(12345)
trainIndex <- createDataPartition(y=new_pml_training$classe, p=0.70, list=FALSE)
trainData <- new_pml_training[trainIndex,]
testData <- new_pml_training[-trainIndex,]
```

``` r
library(rpart)
set.seed(12345)
model.rpart <- rpart(classe~., data= trainData)
prediction.rpart <- predict(model.rpart, newdata= testData, type = "class")
confus.rpart <- confusionMatrix(prediction.rpart, testData$classe)
```

    ## [1] "----------Result of rpart() model----------"

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1498  196   69  106   25
    ##          B   42  669   85   86   92
    ##          C   43  136  739  129  131
    ##          D   33   85   98  553   44
    ##          E   58   53   35   90  790
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.722           
    ##                  95% CI : (0.7104, 0.7334)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6467          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8949   0.5874   0.7203  0.57365   0.7301
    ## Specificity            0.9060   0.9357   0.9097  0.94717   0.9509
    ## Pos Pred Value         0.7909   0.6869   0.6273  0.68020   0.7700
    ## Neg Pred Value         0.9559   0.9043   0.9390  0.91897   0.9399
    ## Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
    ## Detection Rate         0.2545   0.1137   0.1256  0.09397   0.1342
    ## Detection Prevalence   0.3218   0.1655   0.2002  0.13815   0.1743
    ## Balanced Accuracy      0.9004   0.7615   0.8150  0.76041   0.8405

``` r
library(randomForest)
set.seed(12345)
model.rf <- randomForest(classe~.,data= trainData)
prediction.rf <- predict(model.rf, newdata = testData)
confus.rf <- confusionMatrix(prediction.rf, testData$classe)
```

    ## [1] "----------Result of randonForest() model----------"

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    9    0    0    0
    ##          B    1 1127   10    0    0
    ##          C    0    3 1016   13    0
    ##          D    0    0    0  951    5
    ##          E    0    0    0    0 1077
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.993          
    ##                  95% CI : (0.9906, 0.995)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9912         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9895   0.9903   0.9865   0.9954
    ## Specificity            0.9979   0.9977   0.9967   0.9990   1.0000
    ## Pos Pred Value         0.9946   0.9903   0.9845   0.9948   1.0000
    ## Neg Pred Value         0.9998   0.9975   0.9979   0.9974   0.9990
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1915   0.1726   0.1616   0.1830
    ## Detection Prevalence   0.2858   0.1934   0.1754   0.1624   0.1830
    ## Balanced Accuracy      0.9986   0.9936   0.9935   0.9927   0.9977

``` r
library(e1071)
set.seed(12345)
model.svm <- svm(classe~ ., data = trainData)
prediction.svm <- predict(model.svm, testData)
confus.svm <- confusionMatrix(prediction.svm,testData$classe)
```

    ## [1] "----------Result of svm() model----------"

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1670  108    1    2    0
    ##          B    1  994   45    3    7
    ##          C    3   33  961   93   18
    ##          D    0    1   16  864   29
    ##          E    0    3    3    2 1028
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9375         
    ##                  95% CI : (0.931, 0.9435)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9207         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9976   0.8727   0.9366   0.8963   0.9501
    ## Specificity            0.9736   0.9882   0.9697   0.9907   0.9983
    ## Pos Pred Value         0.9377   0.9467   0.8673   0.9495   0.9923
    ## Neg Pred Value         0.9990   0.9700   0.9864   0.9799   0.9889
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2838   0.1689   0.1633   0.1468   0.1747
    ## Detection Prevalence   0.3026   0.1784   0.1883   0.1546   0.1760
    ## Balanced Accuracy      0.9856   0.9304   0.9532   0.9435   0.9742

6. Model selection
==================

From the accuracy in confusionMatrix, the randomForest() accuracy value is 0.9932 while svm() is 0.9375 and rpart() is 0.7720. In this project, we decide to use ramdomForest() model.

    - RandomForest()  accuracy = 0.9932
    - Svm() accuracy = 0.9375
    - Rpart() accuracy = 0.7220

However, the accuracy and sensitively value of randomForest() model can calculate as follows

    - Total accuracy = (1673+1127+1017+950+1077)/(total case = 5885) = 0.9932 (99.32%)
    - Sensitively class A = (1673)/(1673+1) = 0.9994 (99.94%)
    - Sensitively class B = (1127)/(1127+9+3) = 0.9895 (98.95%)
    - Sensitively class C = (1016)/(1016+10) = 0.9903 (99.03%)
    - Sensitively class D = (951)/(950+13) = 0.9865 (98.65%)
    - Sensitively class E = (1077)/(1077+5) = 0.9954 (99.54%) 

7. Prediction
=============

Use randomForest() model to predict 20 value by use pml\_testing

    ## [1] "----- Predict 20 value use ramdomForest() -----"

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
