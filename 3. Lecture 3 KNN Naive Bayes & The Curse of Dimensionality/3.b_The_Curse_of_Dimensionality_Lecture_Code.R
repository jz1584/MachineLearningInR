#######################################################
#######################################################
#####[07] The Curse of Dimensionality Lecture Code#####
#######################################################
#######################################################



#######################
#####Tools for PCA#####
#######################
library(psych) #Library that contains helpful PCA functions, such as:

principal() #Performs principal components analysis with optional rotation.
fa.parallel() #Creates scree plots with parallell analyses for choosing K.
factor.plot() #Visualizes the principal component loadings.


############################
#####Data for Example 1#####
############################
bodies = Harman23.cor$cov #Covariance matrix of 8 physical measurements on 305 girls.
bodies



####################
#####Choosing K#####
####################
fa.parallel(bodies, #The data in question.
            n.obs = 305, #Since we supplied a covaraince matrix, need to know n.
            fa = "pc", #Display the eigenvalues for PCA.
            n.iter = 100) #Number of simulated analyses to perform.
abline(h = 1) #Adding a horizontal line at 1.

#1. Kaiser-Harris criterion suggests retaining PCs with eigenvalues > 1; PCs with
#   eigenvalues < 1 explain less varaince than contained in a single variable.
#2. Cattell Scree test visually inspects the elbow graph for diminishing return;
#   retain PCs before a drastic drop-off.
#3. Run simulations and extract eigenvalues from random data matrices of the same
#   dimension as your data; find where the parallel analysis overshadows real data.



########################
#####Performing PCA#####
########################
pc_bodies = principal(bodies, #The data in question.
                      nfactors = 2, #The number of PCs to extract.
                      rotate = "none")
pc_bodies

#-PC columns contain loadings; correlations of the observed variables with the PCs.
#-h2 column displays the component comunalities; amount of variance explained by
# the components.
#-u2 column is the uniqueness (1 - h2); amount of varaince NOT explained by the
# components.
#-SS loadings row shows the eigenvalues of the PCs; the standardized varaince.
#-Proportion/Cumulative Var row shows the variance explained by each PC.
#-Proportion Explained/Cumulative Proportion row considers only the selected PCs.



########################################
#####Visualizing & Interpreting PCA#####
########################################
factor.plot(pc_bodies,
            labels = colnames(bodies)) #Add variable names to the plot.

#-PC1 correlates highly positively with length-related variables (height, arm
# span, forearm, and lower leg). This is a "length" dimension.
#-PC2 correlates highly positively with volume-related variables (weight, bitro
# diameter, chest girth, and chest width). This is a "volume" dimension.



############################
#####Data for Example 2#####
############################
iris_meas = iris[, -5] #Measurements of iris dataset.
iris_meas
plot(iris_meas)



####################
#####Choosing K#####
####################
fa.parallel(iris_meas, #The data in question.
            fa = "pc", #Display the eigenvalues for PCA.
            n.iter = 100) #Number of simulated analyses to perform.
abline(h = 1) #Adding a horizontal line at 1.
#Should extract 1 PC, but let's look at 2.



########################
#####Performing PCA#####
########################
pc_iris = principal(iris_meas, #The data in question.
                    nfactors = 2,
                    rotate = "none") #The number of PCs to extract.
pc_iris

factor.plot(pc_iris,
            labels = colnames(iris_meas)) #Add variable names to the plot.

#-PC1 separates out the importance of the sepal width as cotrasted with the
# remaining variables.
#-PC2 contrasts the differences between the sepal and petal measurements.



################################
#####Viewing Projected Data#####
################################
plot(iris_meas) #Original data: 4 dimensions.
plot(pc_iris$scores) #Projected data: 2 dimensions.



############################
#####Data for Example 3#####
############################
library(Sleuth2)
case1701
printer_data = case1701[, 1:11]

fa.parallel(printer_data, #The data in question.
            fa = "pc", #Display the eigenvalues for PCA.
            n.iter = 100) #Number of simulated analyses to perform.
abline(h = 1) #Adding a horizontal line at 1.
#Should extract 1 PC, but let's look at 3.

pc_printer = principal(printer_data, #The data in question.
                       nfactors = 3,
                       rotate = "none") #The number of PCs to extract.
pc_printer

factor.plot(pc_printer) #Add variable names to the plot.

#-PC1 ends up being a weighted average.
#-PC2 contrasts one side of the rod with the other.
#-PC3 contrasts the middle of the rod with the sides of the rod.

plot(printer_data)
pairs(pc_printer$scores)



##########################
#####Ridge Regression#####
##########################
library(ISLR)
Hitters = na.omit(Hitters)
help(Hitters)

#Need matrices for glmnet() function. Automatically conducts conversions as well
#for factor variables into dummy variables.
x = model.matrix(Salary ~ ., Hitters)[, -1] #Dropping the intercept column.
y = Hitters$Salary

#Values of lambda over which to check.
grid = 10^seq(5, -2, length = 100)

#Fitting the ridge regression. Alpha = 0 for ridge regression.
library(glmnet)
ridge.models = glmnet(x, y, alpha = 0, lambda = grid)

dim(coef(ridge.models)) #20 different coefficients, estimated 100 times --
#once each per lambda value.
coef(ridge.models) #Inspecting the various coefficient estimates.

#What do the estimates look like for a smaller value of lambda?
ridge.models$lambda[80] #Lambda = 0.2595.
coef(ridge.models)[, 80] #Estimates not close to 0.
sqrt(sum(coef(ridge.models)[-1, 80]^2)) #L2 norm is 136.8179.

#What do the estimates look like for a larger value of lambda?
ridge.models$lambda[15] #Lambda = 10,235.31.
coef(ridge.models)[, 15] #Most estimates close to 0.
sqrt(sum(coef(ridge.models)[-1, 15]^2)) #L2 norm is 7.07.

#Visualizing the ridge regression shrinkage.
plot(ridge.models, xvar = "lambda", label = TRUE, main = "Ridge Regression")

#Can use the predict() function to obtain ridge regression coefficients for a
#new value of lambda, not necessarily one that was within our grid:
predict(ridge.models, s = 50, type = "coefficients")

#Creating training and testing sets. Here we decide to use a 70-30 split with
#approximately 70% of our data in the training set and 30% of our data in the
#test set.
set.seed(0)
train = sample(1:nrow(x), 7*nrow(x)/10)
test = (-train)
y.test = y[test]

length(train)/nrow(x)
length(y.test)/nrow(x)

#Let's attempt to fit a ridge regression using some arbitrary value of lambda;
#we still have not yet figured out what the best value of lambda should be!
#We will arbitrarily choose 5. We will now use the training set exclusively.
ridge.models.train = glmnet(x[train, ], y[train], alpha = 0, lambda = grid)
ridge.lambda5 = predict(ridge.models.train, s = 5, newx = x[test, ])
mean((ridge.lambda5 - y.test)^2)

#Here, the MSE is approximately 115,541.

#What would happen if we fit a ridge regression with an extremely large value
#of lambda? Essentially, fitting a model with only an intercept:
ridge.largelambda = predict(ridge.models.train, s = 1e10, newx = x[test, ])
mean((ridge.largelambda - y.test)^2)

#Here, the MSE is much worse at aproximately 208,920.

#Instead of arbitrarily choosing random lambda values and calculating the MSE
#manually, it's a better idea to perform cross-validation in order to choose
#the best lambda over a slew of values.

#Running 10-fold cross validation.
set.seed(0)
cv.ridge.out = cv.glmnet(x[train, ], y[train],
                         lambda = grid, alpha = 0, nfolds = 10)
plot(cv.ridge.out, main = "Ridge Regression\n")
bestlambda.ridge = cv.ridge.out$lambda.min
bestlambda.ridge
log(bestlambda.ridge)

#What is the test MSE associated with this best value of lambda?
ridge.bestlambdatrain = predict(ridge.models.train, s = bestlambda.ridge, newx = x[test, ])
mean((ridge.bestlambdatrain - y.test)^2)

#Here the MSE is lower at approximately 113,173; a further improvement
#on that which we have seen above. With "cv.ridge.out", we can actually access
#the best model from the cross validation without calling "ridge.models.train"
#or "bestlambda.ridge":
ridge.bestlambdatrain = predict.cv.glmnet(cv.ridge.out, s ="lambda.min", newx = x[test, ])
mean((ridge.bestlambdatrain - y.test)^2)



##########################
#####Lasso Regression#####
##########################
#Fitting the lasso regression. Alpha = 1 for lasso regression.
lasso.models = glmnet(x, y, alpha = 1, lambda = grid)

dim(coef(lasso.models)) #20 different coefficients, estimated 100 times --
#once each per lambda value.
coef(lasso.models) #Inspecting the various coefficient estimates.

#What do the estimates look like for a smaller value of lambda?
lasso.models$lambda[80] #Lambda = 0.2595.
coef(lasso.models)[, 80] #Most estimates not close to 0.
sum(abs(coef(lasso.models)[-1, 80])) #L1 norm is 228.1008.

#What do the estimates look like for a larger value of lambda?
lasso.models$lambda[15] #Lambda = 10,235.31.
coef(lasso.models)[, 15] #Estimates all 0.
sum(abs(coef(lasso.models)[-1, 15])) #L1 norm is essentially 0.

#Visualizing the lasso regression shrinkage.
plot(lasso.models, xvar = "lambda", label = TRUE, main = "Lasso Regression")

#Can use the predict() function to obtain lasso regression coefficients for a
#new value of lambda, not necessarily one that was within our grid:
predict(lasso.models, s = 50, type = "coefficients")

#Let's attempt to fit a lasso regression using some arbitrary value of lambda;
#we still have not yet figured out what the best value of lambda should be!
#We will arbitrarily choose 5. We will now use the training set exclusively.
lasso.models.train = glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
lasso.lambda5 = predict(lasso.models.train, s = 5, newx = x[test, ])
mean((lasso.lambda5 - y.test)^2)

#Here, the MSE is approximately 107,660.

#Instead of arbitrarily choosing random lambda values and calculating the MSE
#manually, it's a better idea to perform cross-validation in order to choose
#the best lambda over a slew of values.

#Running 10-fold cross validation.
set.seed(0)
cv.lasso.out = cv.glmnet(x[train, ], y[train],
                         lambda = grid, alpha = 1, nfolds = 10)
plot(cv.lasso.out, main = "Lasso Regression\n")
bestlambda.lasso = cv.lasso.out$lambda.min
bestlambda.lasso
log(bestlambda.lasso)

#What is the test MSE associated with this best value of lambda?
lasso.bestlambdatrain = predict(lasso.models.train, s = bestlambda.lasso, newx = x[test, ])
mean((lasso.bestlambdatrain - y.test)^2)

#This time the MSE is actually higher at approximately 113,636. What happened?



