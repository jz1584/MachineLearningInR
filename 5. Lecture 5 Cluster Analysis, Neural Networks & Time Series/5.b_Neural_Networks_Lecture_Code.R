###########################################
###########################################
#####[14] Neural Networks Lecture Code#####
###########################################
###########################################

# A website where you can tweak and visualize neural networks.
# http://playground.tensorflow.org

###################################
#####Tools for Neural Networks#####
###################################
#Reading in the data and inspecting its contents.
concrete = read.csv("5.b Concrete.csv")
names(concrete)
str(concrete)
summary(concrete)

#We notice that our data range from values of 0 to upwards of 1,000; neural
#networks work best when we account for the differences in our variables and
#scale accordingly such that values are close to 0. Let's define our own
#normalize function as follows:
normalize = function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

#We now apply our normalization function to all the variables within our dataset;
#we store the result as a data frame for future manipulation.
concrete_norm = as.data.frame(lapply(concrete, normalize))

#Inspecting the output to ensure that the range of each variable is now between
#0 and 1.
summary(concrete_norm)

#Since the data has already been organized in random order, we can simply split
#our data into training and test sets based on the indices of the data frame;
#here we create a 75% training set and 25% testing set split.
concrete_train = concrete_norm[1:773, ]
concrete_test = concrete_norm[774:1030, ]

#Verifying that the split has been successfully made into 75% - 25% segments.
nrow(concrete_train)/nrow(concrete_norm)
nrow(concrete_test)/nrow(concrete_norm)

#Loading the neuralnet library for the training of neural networks.
# install.packages('neuralnet')
library(neuralnet)

#Training the simplest multilayer feedforward neural network that includes only
#one hidden node.
set.seed(0)
concrete_model = neuralnet(strength ~ cement + slag +     #Cannot use the shorthand
                             ash + water + superplastic + #dot (.) notation.
                             coarseagg + fineagg + age,
                           hidden = 1, #Default number of hidden neurons.
                           data = concrete_train)

#Visualizing the network topology using the plot() function.
plot(concrete_model)

#Generating model predictions on the testing dataset using the compute()
#function.
model_results = compute(concrete_model, concrete_test[, 1:8])

#The model_results object stores the neurons for each layer in the network and
#also the net.results which stores the predicted values; obtaining the
#predicted values.
predicted_strength = model_results$net.result

#Examining the correlation between predicted and actual values.
cor(predicted_strength, concrete_test$strength)
plot(predicted_strength, concrete_test$strength)

#Attempting to fit a more complex neural network topology with 5 hidden neurons;
#takes about 10 seconds to run.
set.seed(0)
concrete_model2 = neuralnet(strength ~ cement + slag +
                              ash + water + superplastic +
                              coarseagg + fineagg + age,
                            hidden = 5,
                            data = concrete_train)

#Visualizing the network topology using the plot() function.
plot(concrete_model2)

#Generating model predictions on the testing dataset using the compute()
#function; obtaining the predicted values.
model_results2 = compute(concrete_model2, concrete_test[, 1:8])
predicted_strength2 = model_results2$net.result

#Evaluating the model performance on the test set.
cor(predicted_strength2, concrete_test$strength)
plot(predicted_strength2, concrete_test$strength)

#Can create even more complex models by varying the hidden parameter.
set.seed(0)
concrete_model3 = neuralnet(strength ~ cement + slag +
                             ash + water + superplastic +
                             coarseagg + fineagg + age,
                           hidden = c(15, 3, 4),
                           data = concrete_train)

#Visualizing the network topology using the plot() function.
plot(concrete_model3)

#Generating model predictions on the testing dataset using the compute()
#function; obtaining the predicted values.
model_results3 = compute(concrete_model3, concrete_test[, 1:8])
predicted_strength3 = model_results3$net.result

#Evaluating the model performance on the test set.
cor(predicted_strength3, concrete_test$strength)
plot(predicted_strength3, concrete_test$strength)

#Model 1:
#-1 hidden layer.
#-1 hidden node.
#-Error of 5.082.
#-1,663 steps.
#-Correlation of 0.806.

#Model 2:
#-1 hidden layer.
#-5 hidden nodes.
#-Error of 1.612.
#-25,383 steps.
#-Correlation of 0.928.

#Model 3:
#-3 hidden layers.
#-15, 3, and 4 hidden nodes.
#-Error of 0.709.
#-9,239 steps.
#-Correlation of 0.951.