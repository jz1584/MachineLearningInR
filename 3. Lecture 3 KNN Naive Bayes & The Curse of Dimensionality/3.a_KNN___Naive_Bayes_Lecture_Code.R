#############################
#####K-Nearest Neighbors#####
#############################

# The Stock Market Data
library(ISLR)
## Daily percentage returns for the S&P 500 stock index between 2001 and 2005.
names(Smarket)
dim(Smarket)
summary(Smarket)
pairs(Smarket)
cor(Smarket[,-9])

library(class)
## take all the data before 2005 as training data
train <- Smarket$Year<2005
## split data into training dataset and test dataset
train.X <- cbind(Smarket$Lag1,Smarket$Lag2)[train,]
test.X <- cbind(Smarket$Lag1,Smarket$Lag2)[!train,]
## labels for training dataset
train.Direction <- Smarket$Direction[train]
## labels for test dataset
Direction.2005 <- Smarket$Direction[!train]

set.seed(1)
## perform knn with k = 1
## performance on training dataset
knn.pred=knn(train.X,train.X,train.Direction,k=3)
knn.pred=knn(train.X,train.X,train.Direction,k=1)
table(knn.pred,train.Direction)
mean(knn.pred==train.Direction)
## performance on test dataset
knn.pred <- knn(train.X, test.X, train.Direction,k=1)
table(knn.pred, Direction.2005)
mean(knn.pred == Direction.2005)

## perform KNN with k = 3
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)
mean(knn.pred==Direction.2005)

## with k from 1 to 50
train.acc <- numeric(0)
test.acc <- numeric(0)
k <- seq(50)
for(i in k) {
    knn.pred=knn(train.X,train.X,train.Direction,k=i)
    train.acc[i] <- mean(knn.pred==train.Direction)
    knn.pred=knn(train.X,test.X,train.Direction,k=i)
    test.acc[i] <- mean(knn.pred==Direction.2005)
}
matplot(k, cbind(train.acc, test.acc), type = "l", col = c("red", "blue"), lty = 1)
legend(22,1,c("training accuracy", "test accuracy"), col = c("red", "blue"), lty = 1)


########################################
#####k-Nearest Neighbour Imputation#####
########################################

#K-Nearest Neighbors classification on the iris dataset.
help(iris) #Inspecting the iris measurement dataset.
iris

iris.example = iris[, c(1, 2, 5)] #For illustration purposes, pulling only the
#sepal measurements and the flower species.

#Throwing some small amount of noise on top of the data for illustration
#purposes; some observations are on top of each other.
set.seed(0)
iris.example$Sepal.Length = jitter(iris.example$Sepal.Length, factor = .5)
iris.example$Sepal.Width = jitter(iris.example$Sepal.Width, factor= .5)

col.vec = c(rep("red", 50), #Creating a color vector for plotting purposes.
            rep("green", 50),
            rep("blue", 50))

plot(iris.example$Sepal.Length, iris.example$Sepal.Width,
     col = col.vec, pch = 16,
     main = "Sepal Measurements of Iris Data")
legend("topleft", c("Setosa", "Versicolor", "Virginica"),
       pch = 16, col = c("red", "green", "blue"), cex = .75)

missing.vector = c(41:50, 91:100, 141:150) #Inducing missing values on the Species
iris.example$Species[missing.vector] = NA  #vector for each category.
iris.example

col.vec[missing.vector] = "purple" #Creating a new color vector to
#mark the missing values.

plot(iris.example$Sepal.Length, iris.example$Sepal.Width,
     col = col.vec, pch = 16,
     main = "Sepal Measurements of Iris Data")
legend("topleft", c("Setosa", "Versicolor", "Virginica", "NA"),
       pch = 16, col = c("red", "green", "blue", "purple"), cex = .75)

#Inspecting the Voronoi tesselation for the complete observations in the iris
#dataset.
library(deldir) #Load the Delaunay triangulation and Dirichelet tesselation library.
info = deldir(iris.example$Sepal.Length[-missing.vector],
              iris.example$Sepal.Width[-missing.vector])
plot.tile.list(tile.list(info),
               fillcol = col.vec[-missing.vector],
               main = "Iris Voronoi Tessellation\nDecision Boundaries")

#Adding the observations that are missing species information.
points(iris.example$Sepal.Length[missing.vector],
       iris.example$Sepal.Width[missing.vector],
       pch = 16, col = "white")
points(iris.example$Sepal.Length[missing.vector],
       iris.example$Sepal.Width[missing.vector],
       pch = "?", cex = .66)

library(VIM) #For the visualization and imputation of missing values.

#Conducting a 1NN classification imputation.
iris.imputed1NN = kNN(iris.example, k = 1)

#Assessing the results by comparing to the truth known by the original dataset.
table(iris$Species, iris.imputed1NN$Species)

#Conducting a 12NN classification imputation based on the square root of n.
sqrt(nrow(iris.example))
iris.imputed12NN = kNN(iris.example, k = 12)

#Assessing the results by comparing to the truth known by the original dataset.
table(iris$Species, iris.imputed12NN$Species)


##################################################
#####Using Minkowski Distance Measures in KNN#####
##################################################

library(kknn) #Load the weighted knn library.

#Separating the complete and missing observations for use in the kknn() function.
complete = iris.example[-missing.vector, ]
missing = iris.example[missing.vector, -3]

#Distance corresponds to the Minkowski power.
iris.euclidean = kknn(Species ~ ., complete, missing, k = 12, distance = 2)
summary(iris.euclidean)

iris.manhattan = kknn(Species ~ ., complete, missing, k = 12, distance = 1)
summary(iris.manhattan)


###############################
#####Tools for Naïve Bayes#####
###############################
#Reading in the raw SMS data into a data frame; ensuring that the strings
#aren't converted to factors.
sms_raw = read.csv("[13] SMSSpam.csv", stringsAsFactors = FALSE)

#Examining the structure of the sms data; two columns, one of the actual text itself
#and one displaying whether or not the observation is spam.
str(sms_raw)

#Overwriting the type variable to convert it to a factor.
sms_raw$type = as.factor(sms_raw$type)

#Inspecting the new type variable.
str(sms_raw$type)
table(sms_raw$type)

#Installing the Text Mining package for the purpose of processing text data
#for analysis.
library(tm)

#Creating a corpus with the text message data; VectorSource() interprets each
#element of the vector that it is passed as an individual document.
sms_corpus = Corpus(VectorSource(sms_raw$text))

#Examining the overall contents of the SMS corpus.
print(sms_corpus)

#Examining the specific contents of the SMS corpus; converting the plain text
#documents to character strings.
inspect(sms_corpus[1:3])
lapply(sms_corpus[1:3], as.character)

#Cleaning up the SMS corpus by performing transformations of the text data; first
#convert all letters to lowercase. Using the wrapper function content_transformer().
#Mappings are transformations to corpora.
sms_corpus_clean = tm_map(sms_corpus, content_transformer(tolower))
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

#Looking at the built-in cleaning functions.
getTransformations()

#Removing numbers from the SMS corpus.
sms_corpus_clean = tm_map(sms_corpus_clean, removeNumbers)

#Removing stop words (e.g., to, and, but, or, etc.); can be any list of words.
stopwords()
sms_corpus_clean = tm_map(sms_corpus_clean, removeWords, stopwords())

#Removing punctuation using the built in function.
sms_corpus_clean = tm_map(sms_corpus_clean, removePunctuation)

#Could create a custom function to replace punctuation with a space rather than
#remove it altogether.
removePunctuation("hello...world")
replacePunctuation = function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello...world")

#Using the SnowballC library to performing stemming.
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))

#Stemming the corpus.
sms_corpus_clean = tm_map(sms_corpus_clean, stemDocument)

#Removing the additional whitespace that was left behind when other elements
#were deleted from the corpus.
sms_corpus_clean = tm_map(sms_corpus_clean, stripWhitespace)

#Inspecting the difference between the original text messages and the final
#cleaned text messages.
lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)

#Performing tokenization by creating a document term matrix; a sparse matrix
#with SMS messages as rows and columns as the individual words.
sms_dtm = DocumentTermMatrix(sms_corpus_clean)

#Could also create a sparse document term matrix by passing through a control
#list to the DocumentTermMartix() function; will be slightly different because
#the cleanup functions are applied after they have been split into individual
#words. Different stop words are used; order matters.
sms_dtm2 = DocumentTermMatrix(sms_corpus, control = list(
    tolower = TRUE,
    removeNumbers = TRUE,
    stopwords = TRUE,
    removePunctuation = TRUE,
    stemming = TRUE
))

#To force the two results to be the same, define the stopwords function to
#override the default.
sms_dtm3 = DocumentTermMatrix(sms_corpus, control = list(
    tolower = TRUE,
    removeNumbers = TRUE,
    stopwords = function(x) { removeWords(x, stopwords()) },
    removePunctuation = TRUE,
    stemming = TRUE
))

#Comparing the three final corpus creations.
sms_dtm #Same sms_dtm3.
sms_dtm2 #Different from sms_dtm and sms_dtm3.
sms_dtm3 #Same as sms_dtm.

#Creating training and test sets with a 75% - 25% split; the observations are
#listed in random order.
sms_dtm_train = sms_dtm[1:4169, ]
sms_train_labels = sms_raw[1:4169, ]$type
sms_dtm_test = sms_dtm[4170:5559, ]
sms_test_labels  = sms_raw[4170:5559, ]$type

#Checking that the proportion of spam and non-spam messages is similar among
#the training and test sets.
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

#Loading the wordcloud library to help visualize our corpus data.
library(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50) #Freq. of about 1% of the documents.
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

#Subsetting the data into spam and ham groups.
spam = subset(sms_raw, type == "spam")
ham = subset(sms_raw, type == "ham")

#The wordcloud() function is versatile enough to automatically apply some text
#transformation and tokenization processes to raw data.
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

#Removing terms that are extremely sparse; terms that do not appear in 99.9% of
#the data.
sms_dtm_freq_train = removeSparseTerms(sms_dtm_train, sparse = 0.999)
sms_dtm_train #Before
sms_dtm_freq_train #After

#Displaying indicator features for frequent words (those that appear in at
#least approximately 0.1% of the text messages); saving the terms as a character
#vector.
findFreqTerms(sms_dtm_train, 5)
sms_freq_words = findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)

#Create sparse document term matrices with only the frequent terms.
sms_dtm_freq_train = sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test = sms_dtm_test[, sms_freq_words]

#Since the Naïve Bayes classifier is typically trained on data with categorical
#features, we need to change each of the counts to indicators.
convert_counts = function(x) {
    x = ifelse(x > 0, "Yes", "No")
}

#Using the apply() function to convert the counts to indicators in the columns
#of both the training and the test data.
sms_train = apply(sms_dtm_freq_train, 2, convert_counts)
sms_test = apply(sms_dtm_freq_test, 2, convert_counts)

#Inspecting the final matrices.
head(sms_train)
summary(sms_train)

#Loading the e1071 library in order to implement the Naïve Bayes classifier.
library(e1071)

#Applying the naiveBayes() classifier function to the training data.
sms_classifier = naiveBayes(sms_train, sms_train_labels)
sms_classifier

#Evaluating the model performance by predicting the test observations.
sms_test_pred = predict(sms_classifier, sms_test)
sms_test_pred

#Creating a confusion matrix of the actual and predicted labels.
table(sms_test_pred, sms_test_labels)
(1201 + 153)/1390

#Directly out-of-the-box, the Naive Bayes classifier performs extremely well,
#even when the assumptions are quite unrealistic. We only have an error rate
#of about 2.6%!

#Applying the Laplace estimator and inspecting the accuracy.
sms_classifier2 = naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 = predict(sms_classifier2, sms_test)
table(sms_test_pred2, sms_test_labels)
(1202 + 155)/1390

#Using the Laplace estimator, the error rate decreases slightly to about 2.4%;
#there was a slight reduction in both types of errors.
