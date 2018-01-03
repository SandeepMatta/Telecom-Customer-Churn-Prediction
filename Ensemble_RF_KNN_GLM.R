
#Loading the required libraries
library('caret')
library(randomForest)
library(e1071)
library(Metrics)


#Seeting the random seed
set.seed(1)

data_processed<-read.csv("C:/Users/psneh/Desktop/Sem1/OPIM 5604/project/WA_Fn-UseC_-Telco-Customer-Churn_no_null.csv")

sum(is.na(data_processed))

data_processed$churn_number <- 0
data_processed$churn_number[data_processed$Churn == 'Yes'] <- 1

#covert categorical variables to factor
data_processed$gender <- as.factor(data_processed$gender)
data_processed$SeniorCitizen <- as.factor(data_processed$SeniorCitizen)
data_processed$Partner <- as.factor(data_processed$Partner)
data_processed$Dependents <- as.factor(data_processed$Dependents)
data_processed$PhoneService <- as.factor(data_processed$PhoneService)
data_processed$MultipleLines <- as.factor(data_processed$MultipleLines)
data_processed$InternetService <- as.factor(data_processed$InternetService)
data_processed$OnlineSecurity <- as.factor(data_processed$OnlineSecurity)
data_processed$OnlineBackup <- as.factor(data_processed$OnlineBackup)
data_processed$DeviceProtection <- as.factor(data_processed$DeviceProtection)
data_processed$TechSupport <- as.factor(data_processed$TechSupport)
data_processed$StreamingTV <- as.factor(data_processed$StreamingTV)
data_processed$StreamingMovies <- as.factor(data_processed$StreamingMovies)
data_processed$Contract <- as.factor(data_processed$Contract)
data_processed$PaperlessBilling <- as.factor(data_processed$PaperlessBilling)
data_processed$PaymentMethod <- as.factor(data_processed$PaymentMethod)



#Removing unwanted columns
data_processed <- data_processed[-c(1,21)]
colnames(data_processed)

#Creating Dummies
trainDummy <- dummyVars("~.", data = data_processed, fullRank = F)
train <- as.data.frame(predict(trainDummy,data_processed))
colnames(train)

#Coverting target variable to a factor
train$churn_number <- as.factor(ifelse(train$churn_number == 1,'yes','no'))



#split the date in to train and test
ind <- sample(2,nrow(train), replace = T,prob = c(0.75,0.25))
trainDF<- train[ind ==1,]
testDF <- train[ind ==2,]

#Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = 'final',
  classProbs = T)



#Choose the significant variables in training and testing set
training_set <- trainDF[,c(9,16,18,27,36,39,43,47)]
testing_set <- testDF[,c(9,16,18,27,36,39,43,47)]

#colnames(training_set)

#Defining the predictors and outcome
predictors<-c("tenure","InternetService.Fiber optic","OnlineSecurity.No","TechSupport.No",
              "Contract.Month-to-month","PaperlessBilling.No","PaymentMethod.Electronic check")
outcomeName<-"churn_number"



#1--Training the random forest model
model_rf<-train(training_set[,predictors],training_set[,outcomeName],method='rf',trControl=fitControl,tuneLength=3)

testing_set$pred_rf<-predict(object = model_rf,testing_set[,predictors])

colnames(testing_set)
head(testing_set)

confusionMatrix(testing_set$churn_number,testing_set$pred_rf)

## completion of Random forest


#2--Training the knn model

model_knn<-train(training_set[,predictors],training_set[,outcomeName],
                 method='knn',trControl=fitControl,tuneLength=3)
#Predicting using knn model
testing_set$pred_knn<-predict(object = model_knn,testing_set[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testing_set$churn_number,testing_set$pred_knn)




#3--Training the Logistic regression model

model_lr<-train(training_set[,predictors],training_set[,outcomeName],method='glm',
                trControl=fitControl,tuneLength=3)

#Predicting using Logistic regression model
testing_set$pred_lr<-predict(object = model_lr,testing_set[,predictors])

#Checking the accuracy of Logistic regression model

confusionMatrix(testing_set$churn_number,testing_set$pred_lr)


#Predicting the probabilities
testing_set$pred_rf_prob<-predict(object = model_rf,testing_set[,predictors],type='prob')
testing_set$pred_knn_prob<-predict(object = model_knn,testing_set[,predictors],type='prob')
testing_set$pred_lr_prob<-predict(object = model_lr,testing_set[,predictors],type='prob')

#Taking average of predictions
testing_set$pred_avg<-(testing_set$pred_rf_prob$yes+testing_set$pred_knn_prob$yes+
                         testing_set$pred_lr_prob$yes)/3

testing_set$pred_avg.1<-testing_set$pred_avg

#Splitting into binary classes at 0.5
testing_set$pred_avg<-as.factor(ifelse(testing_set$pred_avg>0.5,'yes','no'))



#Taking weighted average of predictions
testing_set$pred_weighted_avg<-(testing_set$pred_rf_prob$yes*0.25)+(testing_set$pred_knn_prob$yes*0.25)+(testing_set$pred_lr_prob$yes*0.5)
testing_set$pred_weighted_avg.1<-testing_set$pred_weighted_avg

#Splitting into binary classes at 0.5
testing_set$pred_weighted_avg<-as.factor(ifelse(testing_set$pred_weighted_avg>0.5,'yes','no'))

head(testing_set)
# Accuracy of ensembled model
misClasificError <- mean(testing_set$pred_avg != testing_set$churn_number)
print(paste('Accuracy',1-misClasificError))


misClasificError.1 <- mean(testing_set$pred_weighted_avg != testing_set$churn_number)
print(paste('Accuracy',1-misClasificError.1))


#Converting output variable to numeric to calculate AUC
testing_set$churn_number <- as.numeric(ifelse(testing_set$churn_number == 'yes',1,0))


# AUC of individual algorithm and ensembled model
rf_auc<-auc(testing_set$churn_number,testing_set$pred_rf)
print(paste('AUC of RF',rf_auc))

knn_auc<-auc(testing_set$churn_number,testing_set$pred_knn)
print(paste('AUC of KNN',knn_auc))

glm_auc<-auc(testing_set$churn_number,testing_set$pred_lr)
print(paste('AUC of GLM',glm_auc))


ensemble_auc<-auc(testing_set$churn_number,testing_set$pred_avg.1)
print(paste('AUC of ensembled model',ensemble_auc))

ensemble_auc.weighted_av<-auc(testing_set$churn_number,testing_set$pred_weighted_avg.1)
print(paste('AUC of ensembled model',ensemble_auc.weighted_av))


