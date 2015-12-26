#####################  Performing OCR with SVMs  ####################

#### Step 1 - collecting data ####
## Repository (http://archive.ics.uci.edu/ml) ##


### Step 2 - exploring and preparing the data ###
letters <- read.csv("E:\\NuralTechSoft\\Machine Learning\\Machine learning examples\\Machine Learning with R\\Machine-Learning-with-R-datasets-master\\letterdata.csv")
str(letters)

# Given that the data preparation has been largely done for us, we can skip directly to
# the training and testing phases of the machine learning process.
letters_train <- letters[1:16000, ]
letters_test <- letters[16001:20000, ]

# With our data ready to go, let's start building our classifier.
 
### Step 3 - training a model on the data ###
# To provide a baseline measure of SVM performance, let's begin by training a simple
# linear SVM classifier. If you haven't already, install the kernlab package to your library using the command 
# install.packages("kernlab")

library(kernlab)
letter_classifier <- ksvm(letter ~ ., data = letters_train,
                            kernel = "vanilladot")


letter_classifier

#### Step 4 - evaluating model performance ####

letter_predictions <- predict(letter_classifier, letters_test)

# Because we didn't specify the type parameter, the default type = "response" was
# used. This returns a vector containing a predicted letter for each row of values in the
# testing data. Using the head() function, we can see that the first six predicted letters
# were U, N, V, X, N, and H:
head(letter_predictions)

table(letter_predictions, letters_test$letter)


# The following command returns a vector of TRUE or FALSE values indicating
# whether the model's predicted letter agrees with (that is, matches) the actual
# letter in the test dataset:
agreement <- letter_predictions == letters_test$letter

# Using the table() function, we see that the classifier correctly identified the letter in 3,357 out of the 4,000 test records:
table(agreement)

# In percentage terms, the accuracy is about 84 percent:
prop.table(table(agreement))
# Note that when Frey and Slate published the dataset in 1991, they reported a recognition accuracy of about 80 percent.

##### Step 5 - improving model performance ######

# A popular convention is to begin with the Gaussian RBF kernel, which has been shown
# to perform well for many types of data. We can train an RBF-based SVM using the
# ksvm() function as shown here:
letter_classifier_rbf <- ksvm(letter ~ ., data = letters_train, kernel = "rbfdot")

#From there, we make predictions as before:
letter_predictions_rbf <- predict(letter_classifier_rbf,letters_test)

# Finally, we'll compare the accuracy to our linear SVM:
agreement_rbf <- letter_predictions_rbf == letters_test$letter
table(agreement_rbf)

prop.table(table(agreement_rbf))

#################### END ###################
