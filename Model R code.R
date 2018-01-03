setwd("C:/Users/sandeep/OneDrive for Business/PM/Group Project")
churn_data<-read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
library(woe)
library(car)
require("randomForest")
require("ROCR")
require(neuralnet)
require(nnet)
require(ggplot2)
library(caret)
library(Metrics)
library(gbm)

#to get structure of data
str(churn_data)
#to check data 
summary(churn_data)


# imputation 

churn_data$TotalCharges[is.na(churn_data$TotalCharges)]<-0

churn_data$Churn<-ifelse(churn_data$Churn=="Yes",1,0)
churn_data<-churn_data[churn_data$PhoneService=="Yes",]
churn_data<-churn_data[churn_data$OnlineBackup!="No internet service",]

quantile(churn_data$tenure,probs = c(0,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,1))
quantile(churn_data$MonthlyCharges,probs = c(0,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,1))
quantile(churn_data$TotalCharges,probs = c(0,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,1),na.rm = TRUE)

xtabs(~churn_data$gender+churn_data$SeniorCitizen)

set.seed(1234)
indexes = sample(1:nrow(churn_data), size=0.2*nrow(churn_data))
test<-churn_data[indexes,]
train<-churn_data[-indexes,]

rownames(train) <- seq(length=nrow(train))
iv.mult(train,"Churn",TRUE) 

model <- glm(Churn ~ Contract+tenure+OnlineSecurity+TotalCharges+InternetService+PaymentMethod+TechSupport,
             family=binomial(link='logit'),data=train)

summary(model)
vif(model)

train.results <- predict(model,newdata=train,type='response')
train.results <- ifelse(train.results > 0.50,1,0)
misClasificError <- mean(train.results != train$Churn)
print(paste('Accuracy',1-misClasificError))

fitted.results <- predict(model,newdata=test,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test$Churn)
print(paste('Accuracy',1-misClasificError))

# RF

model.rf <- randomForest(as.factor(Churn) ~ ., data=train[,-1], importance=TRUE,
                         proximity=TRUE,ntree=(100),nodesize=100)
importance(model.rf)

train$prob_rf<-predict(model.rf,type="prob",newdata = train)[,2]
test$prob_rf<-predict(model.rf,type="prob",newdata = test)[,2]

print(auc(train$Churn,train$prob_rf))
print(auc(test$Churn,test$prob_rf))
fitted.results<-ifelse(test$prob_rf>0.4,1,0)
misClasificError <- mean(fitted.results != test$cancel)
print(paste('Accuracy for',i*100,'tree is',1-misClasificError))


## inicator variables

churn_data$Churn_ind<-ifelse(churn_data$Churn=="Yes",1,0)
churn_data$Churn<-NULL
str(churn_data)
nums<-colnames(churn_data[,sapply(1:ncol(churn_data),function(x) is.numeric(churn_data[,x]))])
chars<-colnames(churn_data[,sapply(1:ncol(churn_data),function(x) is.factor(churn_data[,x]))])
chars<-chars[2:16]

churn_data_chars<-data.frame(sapply(1:length(chars),function(x) class.ind(churn_data[,chars[x]])))
churn_data_nums<-churn_data[,nums]

churn_data_ind<-cbind(churn_data_chars,churn_data_nums)

churn_data_ind[,1:42]<-data.frame(sapply(1:42,function(x) as.factor(churn_data_ind[,x])))

set.seed(1234)
indexes = sample(1:nrow(churn_data_ind), size=0.2*nrow(churn_data_ind))
test<-churn_data_ind[indexes,]
train<-churn_data_ind[-indexes,]

rownames(train) <- seq(length=nrow(train))
a<-iv.mult(train,"Churn_ind",TRUE)
b<-a$Variable[1:20]
f<-paste("Churn_ind ~ ",paste(b,collapse = "+"),sep = "")
model <- glm(as.formula(f),
             family=binomial(link='logit'),data=train)

summary(model)
vif(model)
model$formula

model <- glm(Churn_ind ~ Month.to.month + Two.year + tenure + No.5 + No.8 + 
               Fiber.optic + Electronic.check   + 
               No.4  + No.11,
             family=binomial(link='logit'),data=train)

summary(model)
vif(model)

train.results <- predict(model,newdata=train,type='response')
auc(train$Churn_ind,train.results)
train.results <- ifelse(train.results > 0.5,1,0)
auc(train$Churn_ind,train.results)

misClasificError <- mean(train.results != train$Churn)
print(paste('Accuracy',1-misClasificError))

fitted.results <- predict(model,newdata=test,type='response')
auc(test$Churn_ind,fitted.results)
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test$Churn)
print(paste('Accuracy',1-misClasificError))

## Chaid tree

library("CHAID", lib.loc="~/R/win-library/3.4")

chaid.control<-chaid_control(alpha2 = 0.05, alpha3 = -1, alpha4 = 0.05,
                             minsplit = 20, minbucket = 7, minprob = 0.05,
                             stump = FALSE, maxheight = 4)

train$Churn_ind<-as.factor(train$Churn_ind)
chars<-sapply(train,is.factor)

tree<-chaid(as.factor(Churn_ind)~.,train[,chars],control=chaid.control)

# Lift curve and ROC curve

# Plot the performance of the model applied to the evaluation set as
# an ROC curve.

require(ROCR)
pred <- ROCR::prediction(test$prob, test$churn_number)
perf <- performance(pred,"tpr","fpr")
plot(perf, main="ROC curve", colorize=T)

# And then a lift chart
perf <- performance(pred,"lift","rpp")
plot(perf, main="lift curve", colorize=T)


# Learning rate

set.seed(123)
sample = sample.split(churn$churn_number, SplitRatio = .8)
train = subset(churn, sample == TRUE)
test  = subset(churn, sample == FALSE)

fit<- glm(train$churn_number ~ tenure * Contract_MM +
            Internet_FiberOptic+PaperlessBilling_Yes+Electronic_check_PM,
          family = binomial("logit"),
          data = train)

train$prob<-predict(fit,newdata=train,type='response')

fit.result.train<-ifelse(train$prob > 0.5,1,0)
misClasificError.train <- mean(fit.result.train != train$churn_number)
print(paste('Accuracy',1-misClasificError.train))
xtab<-  table(train$churn,fit.result.train)
confusionMatrix(xtab)

test$prob <- predict(fit,newdata=test)
fit.results.prob <- ifelse(test$prob > 0.5,1,0)
misClasificError <- mean(fit.results.prob != test$churn_number)
print(paste('Accuracy',1-misClasificError))

train_pt<-c(0.005,0.01,0.1,0.2,0.3,1)
train_accuracy<-c(1,0.816901408450704,0.784090909090909,0.794889992902768,0.793658305726455,0.793658305726455)
train_accuracy<-1-train_accuracy

test_pt<-c(0.005,0.01,0.1,0.2,0.3,1)
test_accuracy<-c(0.696917808219178,0.769650028686173,0.788294683704054,0.789669861554846,0.788032454361055,0.788032454361055)
test_accuracy<-1-test_accuracy

plot(train_pt,train_accuracy,type="l",col="red",ylim=c(0,0.4),xlab = "Training size",ylab="Misclassification")
lines(test_pt,test_accuracy,col="green")
?plot


