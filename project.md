# Predicting correct barbell lifts using wearables data
Pelayo Arbués  
13 de mayo de 2015  

#Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

#Data 

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>.

#Goal

The goal of this project is to predict the manner in which they did the exercise.


#Data downloading

In this first step I set the working directory in order to download the training and test datasets. When reading the data, I have labelled those observations that take blank values and #DIV/0! as NAs.


```r
#Set working directory
setwd("/Users/pelayogonzalez/Desktop/Coursera/Practical_Machine_Learning/Project")

#Create data folder to store dataframes
if(!file.exists("data")){
        dir.create("data")
}

#Download data
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url_train, destfile = "./data/pml-training.csv", method="curl")
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url_test, destfile = "./data/pml-testing.csv", method="curl")

list.files("./data")


#Read training set
training <- read.table("./data/pml-training.csv", 
                       header = TRUE,
                       sep=",",
                       na.strings = c("#DIV/0!", ""))
#Read test set
finaltest <- read.table("./data/pml-testing.csv", 
                      header=TRUE,
                      sep=",",
                      na.strings = c("#DIV/0!", ""))
```

#Pre-processing
After downloading the data we check the dimensions of the dataframe. There are 159 potential explanatory variables (covariates), although many of these variables are filled with NAs. I have also disregarded those variables with near zero variance as they are not likely to be good predictors. Finally, in order to keep reducing the dimensions of the dataframe I have also removed other variables that indicated the ID of the observation, time and date when data was registered or information about the subject. 


```r
#Require packages
packages <- c("ggplot2", "dplyr","caret")
sapply(packages, require, character.only=TRUE, quietly=TRUE)

#Let's remove those variables with a proportion of NAs >90%
dimcols <- dim(training)[2]
training<- training[ , sapply(training, 
                        function(x) !sum(is.na(x))/dimcols>.9)] 

#Using the caret library we are also removing those variables with near zero
#variance
near0 <- nearZeroVar(training,saveMetrics = T)
training <- training[,near0$nzv == FALSE]

#Finally we also remove other variables (ID, dates...)
toMatch <- c("arm", "dumbell", "belt","classe")
training <- training[,grep(paste(toMatch,collapse="|"), 
                            colnames(training))]
```


#Cross-validation and model estimation

In order to avoid overidentification issues, training dataframe is split in trainingSet and validationSet. TrainingSet will be used to train the model while the validationSet partition will be used to estimate the out of sample errors.


```r
#Slicing data for cross-validation
set.seed(1234)
inTrain <- createDataPartition(y=training$classe,
                               p=0.7, 
                               list=FALSE)
trainingSet <- training[inTrain,]
validationSet <- training[-inTrain,]
```

I have also checked the correlation between the 39 remaining covariates.

```r
library(corrplot)
corMatrix<- cor(subset(trainingSet, select = -classe))
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![](project_files/figure-html/unnamed-chunk-4-1.png) 

According to the plot, there is big correlation between some variables, which might suggest the use of Principal Components Analysis. Preprocessing includes the use of cross-validation (4 folds) and Principal Components Analysis use. 

The variable to be explained is a categorical variable and we expect to predict this variable by using other variables that capture different types of movements. The relation between the dependent variable and data features is likely to be non linear so the model to be estimated is Random Forest.


```r
tc <- trainControl(method = "cv", number = 4, verboseIter=FALSE,
                   preProcOptions="pca", allowParallel=TRUE)
#Random forest model estimation
modFit <- train(trainingSet$classe ~.,
                data = trainingSet,
                method = "rf",
                verbose=F, 
                trControl = tc)
```


#Model testing and prediction
In the next chunk we evaluate the performance of the model by calculating the Confusion Matrix and model accuracy.

```r
# 
prediction <- predict(modFit, validationSet)
confusionMatrix(prediction, validationSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    5    0    0    0
##          B    2 1132    5    2    2
##          C    0    2 1013   10    0
##          D    0    0    8  952    4
##          E    0    0    0    0 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9939   0.9873   0.9876   0.9945
## Specificity            0.9988   0.9977   0.9975   0.9976   1.0000
## Pos Pred Value         0.9970   0.9904   0.9883   0.9876   1.0000
## Neg Pred Value         0.9995   0.9985   0.9973   0.9976   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1924   0.1721   0.1618   0.1828
## Detection Prevalence   0.2850   0.1942   0.1742   0.1638   0.1828
## Balanced Accuracy      0.9988   0.9958   0.9924   0.9926   0.9972
```

```r
accur <- postResample(validationSet$classe, prediction)
modAccuracy <- accur[[1]]
modAccuracy
```

```
## [1] 0.9932031
```


```r
out_of_sample_error <- 1 - modAccuracy
out_of_sample_error
```

```
## [1] 0.006796941
```
Out of sample error takes the value 0.0067969 which indicates the good predictive performance of the model.

Finally, we use the trained model to predict the values of the 20 observations required for the assignment using the finaltest dataframe.

```r
rfPred <- predict(modFit, finaltest)
```

Generation of the files to be submitted using the provided code:


```r
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

pml_write_files(rfPred)
```


