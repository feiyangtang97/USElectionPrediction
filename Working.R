setwd("~/Downloads/Jackie/760/new project/ga_data_science_final_project-master")
# Random forests
library(caTools)
voting16 <- read.csv("combined_data.csv")
a = as.factor(voting16$Democrat)
voting16$Democrat = a
vote2016 = voting16[, c(-1, -2)]
voting12 <- read.csv("combined_data2012.csv")
str(voting12)
b = as.factor(voting12$Democrat)
voting12$Democrat = b
vote2012 = voting12[, c(-1, -2)]
split <- sample.split(vote2016, SplitRatio = .8)
training <- subset(vote2016, split =="TRUE")
testing <- subset(vote2016, split =="FALSE")

random1 <- randomForest(Democrat~., data = training)
importance(random1)
varImpPlot(random1)
pre <- predict(random1, testing, type = "class")
t <- table(predictions = pre, actual = testing$Democrat)
t
sum(diag(t))/sum(t)




split12 <- sample.split(vote2012, SplitRatio = .8)
training12 <- subset(vote2012, split12 =="TRUE")
testing12 <- subset(vote2012, split12 =="FALSE")
random2 <- randomForest(Democrat~., data = training12)
importance(random2)
varImpPlot(random2)
pre12 <- predict(random2, testing12, type = "class")
t1 <- table(predictions = pre12, actual = testing12$Democrat)
t1
sum(diag(t1))/sum(t1)
pre16 <- predict(random2, vote2016, type = "class")
t2 <- table(predictions = pre16, actual = vote2016$Democrat)
t2
sum(diag(t2))/sum(t2)
plot(random2)
library(randomForest)
random3 <- randomForest(Democrat~., data = vote2012)
pre16all <- predict(random3, vote2016, type = "class")
t3 <- table(predictions = pre16all, actual = vote2016$Democrat)
t3
sum(diag(t3))/sum(t3)




#ROC
library(pROC)
PredictionsWithProbs <- predict(random3, vote2016, type = "prob")
auc <- auc(vote2016$Democrat, PredictionsWithProbs[,2])
auc
plot(roc(vote2016$Democrat, PredictionsWithProbs[,2]),print.thres = T,
     print.auc=T)

#KNN
voting12L <- read.csv("combined_data2012.csv")
voting16L <- read.csv("combined_data.csv")
vote2012L = voting12L[, c(-1, -2)]
vote2016L = voting16L[, c(-1, -2)]
str(vote2016L)
vote2012L$Zoroastrian = as.numeric(vote2012L$ Zoroastrian) 
normalize <- function (x) {
  return((x - min(x))/ (max(x) - min(x)))}
voting12L_n <- as.data.frame(lapply(vote2012L[,-76], normalize))
voting16L_n <- as.data.frame(sapply(vote2016L[,-76], normalize))
str(voting16L_n)
voting12L_nn = voting12L_n[,-c(23, 31, 51, 65, 69)]
voting16L_nn = voting16L_n[,-c(23, 31, 51, 65, 69)]
str(voting16L_nn)
library(class)
test_target <- voting16L$Democrat
train_target <- voting12L$Democrat
length(train_target)
m1 <- knn(train = voting12L_nn, test = voting16L_nn, cl=train_target, k =9)
t4 = table(test_target, m1)
sum(diag(t4))/sum(t4)

# cross validation
library(mlbench)
library(caret)

voting12L_n1 = voting12L_nn
voting12L_n1$Democrat = as.factor(voting12L$Democrat)
voting16L_n1 = voting16L_nn
voting16L_n1$Democrat = as.factor(voting16L$Democrat)
voting12L_n2 = voting12L_n1
voting12L_n2$Democrat = ifelse(voting12L_n1$Democrat == 1, "Y", "N")
voting16L_n2 = voting16L_n1
voting16L_n2$Democrat = ifelse(voting16L_n1$Democrat == 1, "Y", "N")
control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE)

RF <- train(Democrat ~ .,
             data=vote2012, method="rf", trControl=control)

KNN <- train(Democrat ~ .,
             data=voting12L_n1, method="knn", trControl=control)
ADA <- train(Democrat ~ .,
             data=vote2012, method="AdaBoost.M1", trControl=control)

SVM <- train(Democrat ~ ., 
             data=voting12L_n2, method="svmRadial", trControl=control)


results <- resamples(list(RF = RF, ADA = ADA, KNN = KNN, SVM = SVM))

summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)


scales <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(results, scales=scales, pch = "|", auto.key = list(columns = 4))
diffs <- diff(results)
# summarize p-values for pair-wise comparisons
summary(diffs)

#Prediction
Prerandom <- predict(RF, vote2016)
t6 <- table(predictions = Prerandom, actual = vote2016$Democrat)
t6
sum(diag(t6))/sum(t6)

434/(180+434)
2479/(2479 + 180)
PredictionsWithProbs <- predict(RF, vote2016, type = "prob")
plot(roc(vote2016$Democrat, PredictionsWithProbs[,2]),print.thres = T,
     print.auc=T, main = "ROC curve for RF model")

plot(varImp(RF), top = 10)
plot(varImp(ADA), top = 10)
plot(varImp(KNN), top = 10)
plot(varImp(SVM), top = 10)


PreADA <- predict(ADA, vote2016)
t7 <- table(predictions = PreADA, actual = vote2016$Democrat)
t7
sum(diag(t7))/sum(t7)
2503/(2503 + 156)
405/(405+81)

PredictionsWithProbs1 <- predict(ADA, vote2016, type = "prob")
plot(roc(vote2016$Democrat, PredictionsWithProbs1[,2]),print.thres = T,
     print.auc=T, main = "ROC curve for ADA model")


PreKNN <- predict(KNN, voting16L_n1)
t8 <- table(predictions = PreKNN, actual = voting16L_n1$Democrat)
t8
sum(diag(t8))/sum(t8)


326/(326 +160)
2602/(2602 + 57)
PredictionsWithProbs2 <- predict(KNN, voting16L_n1, type = "prob")
plot(roc(voting16L_n1$Democrat, PredictionsWithProbs2[,2]),print.thres = T,
     print.auc=T, main = "ROC curve for KNN model")



PreSVM <- predict(SVM, voting16L_n1)
t9 <- table(predictions = PreSVM, actual = voting16L_n2$Democrat)
t9
sum(diag(t9))/sum(t9)
2604/(2604 + 55)
321/ (321 + 165)

PredictionsWithProbs3 <- predict(SVM, voting16L_n2, type = "prob")
plot(roc(voting16L_n2$Democrat, PredictionsWithProbs3[,2]),print.thres = T,
     print.auc=T, main = "ROC curve for SVM model")

par(mfrow=c(2,2))





mod <- class::knn(cl = train_target,
                  test = voting16L_nn,
                  train = voting12L_nn,
                  k = 63,
                  prob = TRUE)
roc(test_target, attributes(mod)$prob)
plot(roc(test_target, attributes(mod)$prob),
     print.thres = T,
     print.auc=T)



#Adaboosting
library(adabag)
adaboost<-boosting(Democrat~., data=vote2012, boos=TRUE, mfinal=20,coeflearn='Breiman')
str(vote2012$Democrat)
summary(adaboost)
# Test
ADB <- predict(adaboost,vote2016)
1 - ADB$error
# Validate
error
error = mean(errorevol(adaboost,vote2012)$error)
1-error

#Roc

#ROC
library(pROC)
PredictionsWithProbs1 <- predict(adaboost, vote2016, type = "prob")
auc <- auc(vote2016$Democrat, PredictionsWithProbs1$prob[,2])
auc
plot(roc(vote2016$Democrat, PredictionsWithProbs1$prob[,2]),print.thres = T,
     print.auc=T)


model_list <- list(RandomForests = random3, Adaboost = adaboost, KNN = m1)
res <- resamples(model_list)
summary(random3)
bwplot(model_list)
xyplot(res, metric = "RMSE")






#logistic





