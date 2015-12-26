####################### Modeling the strength of concrete with ANNs ######################

## Step 1 - collecting data ##
# For this analysis, we will utilize data on the compressive strength of concrete donated
# to the UCI Machine Learning Data Repository (http://archive.ics.uci.edu/ml)

## Step 2 - exploring and preparing the data ##
concrete <- read.csv("E:\\NuralTechSoft\\Machine Learning\\Machine learning examples\\Machine Learning with R\\Machine-Learning-with-R-datasets-master\\concrete.csv")
str(concrete)

summary(concrete_norm$strength)

# Typically, the solution to this problem is to rescale the data with a normalizing or
# standardization function.
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))

# To confirm that the normalization worked, we can see that the minimum and
# maximum strength are now 0 and 1, respectively:
summary(concrete_norm$strength)

# In comparison, the original minimum and maximum values were 2.33 and 82.6:
summary(concrete$strength)

concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

### Step 3 - training a model on the data ###

# you will need to install it by typing
# install.packages("neuralnet")

library(neuralnet)
concrete_model <- neuralnet(strength ~ cement + slag +
                              ash + water + superplastic +
                              coarseagg + fineagg + age,
                            data = concrete_train)

plot(concrete_model)

#### Step 4 - evaluating model performance ####
model_results <- compute(concrete_model, concrete_test[1:8])
predicted_strength <- model_results$net.result

# Recall that the cor() function is used to obtain a correlation between two numeric vectors:
cor(predicted_strength, concrete_test$strength)


###### Step 5 - improving model performance ######

concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic +
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden = 5)

plot(concrete_model2)

# Applying the same steps to compare the predicted values to the true values, we
# now obtain a correlation around 0.80, which is a considerable improvement over
# the previous result:
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)


##################### END ###################


