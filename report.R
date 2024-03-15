if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071")
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC")
if (!requireNamespace("skimr", quietly = TRUE)) install.packages("skimr")
if (!requireNamespace("corrplot", quietly = TRUE)) install.packages("corrplot")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("GGally", quietly = TRUE)) install.packages("GGally")

library("caret")
library("e1071")
library("randomForest")
library("pROC")
library("corrplot")
library("skimr")
library("dplyr")
library("GGally")

# Load raw data from URL
data <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")

str(data)

skim(data)

ggplot(data, aes(x = Personal.Loan)) +
  geom_bar() +
  labs(title = "Distribution of Personal Loan Variable", x = "Personal Loan", y = "Count")

numeric_features <- setdiff(names(data)[sapply(data, is.numeric)], "Personal.Loan")

for (feature in numeric_features) {
  # Check if the feature is continuous
  if (length(unique(data[[feature]])) > 10) { # Arbitrary threshold, adjust based on your data
    p <- ggplot(data, aes_string(x = feature)) +
      geom_histogram(bins = 30, fill = "skyblue", color = "black") +
      labs(title = paste(feature, "distribution"), x = feature, y = "frequency")
  } else {
    # For discrete data, use a bar plot
    p <- ggplot(data, aes_string(x = feature)) +
      geom_bar(fill = "orange", color = "black") +
      labs(title = paste(feature, "distribution"), x = feature, y = "count")
  }
  print(p)
}

personal_loan_data = readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
personal_loan_data$Education <- as.numeric(personal_loan_data$Education)
personal_loan_data$Personal.Loan <- as.numeric(personal_loan_data$Personal.Loan)
personal_loan_data$Securities.Account <- as.numeric(personal_loan_data$Securities.Account)
personal_loan_data$CD.Account <- as.numeric(personal_loan_data$CD.Account)
personal_loan_data$Online <- as.numeric(personal_loan_data$Online)
personal_loan_data$CreditCard <- as.numeric(personal_loan_data$CreditCard)

cor_matrix <- cor(personal_loan_data[, sapply(personal_loan_data, is.numeric)], use="complete.obs")
corrplot(cor_matrix, method="color", tl.col="black", tl.srt=45)

personal_loan_data$Education <- as.factor(personal_loan_data$Education)
personal_loan_data$CD.Account <- as.factor(personal_loan_data$CD.Account)
personal_loan_data$Personal.Loan <- as.factor(personal_loan_data$Personal.Loan)

ggplot(personal_loan_data, aes(x = Income, fill = Personal.Loan)) + 
  geom_histogram(binwidth = 10, position = "dodge") + 
  labs(title = "Income Distribution by Personal Loan Acceptance",
       x = "Income",
       y = "Count")

ggplot(personal_loan_data, aes(x = CCAvg, fill = Personal.Loan)) + 
  geom_histogram(binwidth = 1, position = "dodge") +
  labs(title = "CCAvg Distribution by Personal Loan Acceptance", x = "CCAvg", y = "Count")

ggplot(personal_loan_data, aes(x = Mortgage, fill = Personal.Loan)) + 
  geom_histogram(binwidth = 50, position = "dodge") +
  labs(title = "Mortgage Distribution by Personal Loan Acceptance", x = "Mortgage", y = "Count")

ggplot(personal_loan_data, aes(x = CD.Account, fill = Personal.Loan)) + 
  geom_bar(position = "dodge") +
  labs(title = "CD Account Holders by Personal Loan Acceptance", x = "CD Account", y = "Count")

ggplot(personal_loan_data, aes(x = Education, fill = Personal.Loan)) + 
  geom_bar(position = "dodge") +
  labs(title = "Education Level by Personal Loan Acceptance", x = "Education Level", y = "Count")

selected_vars <- personal_loan_data %>% select(Income, CCAvg, Education, Mortgage, CD.Account, Personal.Loan)
ggpairs(selected_vars)

# Set seed value to ensure reproducibility of results.
set.seed(123)

# Data preprocessing.
# Make sure the levels of the factor variables are valid R variable names.
data$Personal.Loan <- factor(data$Personal.Loan, levels = c("0", "1"), labels = c("No", "Yes"))
data$Securities.Account <- factor(data$Securities.Account, levels = c("0", "1"), labels = c("No", "Yes"))
data$CD.Account <- factor(data$CD.Account, levels = c("0", "1"), labels = c("No", "Yes"))
data$Online <- factor(data$Online, levels = c("0", "1"), labels = c("No", "Yes"))
data$CreditCard <- factor(data$CreditCard, levels = c("0", "1"), labels = c("No", "Yes"))

# Divide training set and test set
index <- createDataPartition(data$Personal.Loan, p = 0.8, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Set cross-validation method
control <- trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=twoClassSummary)

# Train logistic regression model
logit_model <- train(Personal.Loan ~ ., data=train_data, method="glm", family="binomial", trControl=control, metric="ROC")

print(logit_model)
summary(logit_model)

# Train support vector machine model
svm_model <- train(Personal.Loan ~ ., data=train_data, method="svmRadial", trControl=control, metric="ROC", preProcess=c("center", "scale"))

print(svm_model)
summary(svm_model)

# Train random forest model
rf_model <- train(Personal.Loan ~ ., data=train_data, method="rf", trControl=control, metric="ROC", ntree=100)

print(rf_model)
summary(rf_model)

# Predict and evaluate the model
predictions <- list(
  logit = predict(logit_model, test_data, type="prob")[, "Yes"],
  svm = predict(svm_model, test_data, type="prob")[, "Yes"],
  rf = predict(rf_model, test_data, type="prob")[, "Yes"]
)

# Calculate ROC curve and AUC
roc_results <- lapply(predictions, function(pred) roc(response = as.numeric(test_data$Personal.Loan) - 1, predictor = pred))

# Calculate and format AUC values
auc_values <- sapply(roc_results, function(x) round(auc(x), 4))

# Format and print AUC value
cat("\nAUC Values for Models:\n")
cat("Logistic Regression: ", auc_values["logit"], "\n")
cat("SVM:                 ", auc_values["svm"], "\n")
cat("Random Forest:       ", auc_values["rf"], "\n")

# Prepare ROC data
roc_data <- data.frame(
  model = factor(c(rep("Logistic Regression", length(roc_results$logit$sensitivities)),
                   rep("SVM", length(roc_results$svm$sensitivities)),
                   rep("Random Forest", length(roc_results$rf$sensitivities))),
                 levels = c("Logistic Regression", "SVM", "Random Forest")),
  specificity = c(1 - roc_results$logit$specificities,
                  1 - roc_results$svm$specificities,
                  1 - roc_results$rf$specificities),
  sensitivity = c(roc_results$logit$sensitivities,
                  roc_results$svm$sensitivities,
                  roc_results$rf$sensitivities)
)

# Draw ROC curve
roc_plot <- ggplot(roc_data, aes(x = specificity, y = sensitivity, color = model)) +
  geom_line(size = 0.6) +
  scale_color_manual(values = c("orange", "skyblue", "#AB545A")) +
  labs(x = "1 - Specificity", y = "Sensitivity", color = "Model", title = "ROC Curves") +
  theme_minimal() +
  theme(
    text = element_text(size = 12),
    legend.position = "bottom",
    legend.title.align = 0.5,
    plot.title = element_text(hjust = 0.5)
  ) +
  coord_fixed(ratio = 1) +
  geom_abline(linetype = "dashed")

# Print ROC curve graph
print(roc_plot)

if (!requireNamespace("mlr3", quietly = TRUE)) install.packages("mlr3")
if (!requireNamespace("mlr3learners", quietly = TRUE)) install.packages("mlr3learners")
if (!requireNamespace("mlr3tuning", quietly = TRUE)) install.packages("mlr3tuning")
if (!requireNamespace("mlr3viz", quietly = TRUE)) install.packages("mlr3viz")
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("precrec", quietly = TRUE)) install.packages("precrec")

library("mlr3")
library("mlr3learners")
library("mlr3tuning")
library("mlr3viz")
library("data.table")
library("ggplot2")
library("precrec")

# new data
data <- fread("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")

# Data preprocessing: Convert the target variable Personal.Loan to factor type
data$Personal.Loan <- as.factor(data$Personal.Loan)

# Create a task
task <- TaskClassif$new(id = "LoanUpsell", backend = data, target = "Personal.Loan")

# Three different classification algorithms (logistic regression, SVM, random forest) were selected for model training and comparison.
learners <- list(
  lrn("classif.log_reg", predict_type = "prob"),
  lrn("classif.svm", predict_type = "prob"),
  lrn("classif.ranger", predict_type = "prob")
)

# Define 10-fold cross-validation to evaluate model performance.
resampling <- rsmp("cv", folds = 5)

# Define evaluation design
design <- benchmark_grid(
  tasks = task,
  learners = learners,
  resamplings = resampling
)

if (!is.data.frame(design)) {
  stop("The design is not a data frame structure.")
}

# Function trains and compares models, aggregating results based on the AUC metric.
bmr <- benchmark(design)
bmr_results <- bmr$aggregate(msr("classif.auc"))

# Visualize the AUC performance of different models.
autoplot(bmr, measure = msr("classif.auc"))

# Create a new classification task task_rf_optimization
task_rf_optimization <- TaskClassif$new(id = "LoanUpsell", backend = data, target = "Personal.Loan")

# Create a learner for a random forest model
# Set predict_type to prob to predict probability
learner_rf <- lrn("classif.ranger", predict_type = "prob")

# Use ParamSet$new to define the parameter set param_set for model tuning
# mtry: Number of variables considered in each split, ranging from sqrt(ncol(data)/3) to sqrt(ncol(data)).
# min.node.size: The minimum number of samples of leaf nodes of the tree, ranging from 1 to 10.
# num.trees: Number of trees in the forest, ranging from 100 to 1000.
param_set <- ParamSet$new(params = list(
  ParamInt$new("mtry", lower = as.integer(sqrt(ncol(data)/3)), upper = as.integer(sqrt(ncol(data)))),
  ParamInt$new("min.node.size", lower = 1, upper = 10),
  ParamInt$new("num.trees", lower = 100, upper = 1000)
))

# A random search tuner is defined, and 10 parameter combinations are randomly selected for evaluation in each batch.
tuner <- tnr("random_search", batch_size = 10)

# Create an automatic tuner at
at <- AutoTuner$new(
  learner = learner_rf,                    # random forest learner
  resampling = rsmp("cv", folds = 5),     # 10-fold cross validation
  measure = msr("classif.auc"),            # AUC performance metric
  tuner = tuner,                           # Random search tuner
  search_space = param_set,                # Parameter search space
  terminator = trm("evals", n_evals = 50)  # Stopping criterion is set to 50 evaluations
)

# Training the autotuner
at$train(task_rf_optimization)

# Performance evaluation and plot roc
prediction <- at$predict(task_rf_optimization)
autoplot(prediction, type = "roc")

# Print confusion matrix conf_mat
conf_mat <- prediction$confusion
print(conf_mat)

# The archive variable stores the detailed results of each evaluation during the tuning process.
archive <- at$archive
print(names(archive$data))

ggplot(archive$data, aes(y = classif.auc)) +
  geom_boxplot() +
  labs(title = "Performance Distribution Across Tuning Iterations",
       y = "AUC Score") +
  theme_minimal()

ggplot(archive$data, aes(x = mtry, y = classif.auc)) +
  geom_point() +
  geom_smooth() +
  labs(title = "mtry vs. AUC", x = "mtry", y = "AUC Score") +
  theme_minimal()

ggplot(archive$data, aes(x = min.node.size, y = classif.auc)) +
  geom_point() +
  geom_smooth() +
  labs(title = "min.node.size vs. AUC", x = "Min. Node Size", y = "AUC Score") +
  theme_minimal()

ggplot(archive$data, aes(x = num.trees, y = classif.auc)) +
  geom_point() +
  geom_smooth() +
  labs(title = "num.trees vs. AUC", x = "Number of Trees", y = "AUC Score") +
  theme_minimal()

if (!requireNamespace("keras", quietly = TRUE)) install.packages("keras")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")

library("keras")
library("tidyverse")
library("data.table")

#Convert it to a factor, and then convert it to a numerical type to adapt to the needs of the deep learning model for processing binary classification problems.
data$Personal.Loan <- as.factor(data$Personal.Loan)
target_variable <- as.numeric(as.character(data$Personal.Loan))

# Select features
features <- data %>% select(-Personal.Loan) %>% as.matrix()
# Standardize features
features_scaled <- scale(features)

set.seed(123)

# Randomly divide the data set into training set and test set.
# 80% of the data is used as the training set.
# Split the training set and test set of features and target variables according to the index.
index <- sample(1:nrow(data), round(0.8 * nrow(data)))
x_train <- features_scaled[index, ]
y_train <- target_variable[index]
x_test <- features_scaled[-index, ]
y_test <- target_variable[-index]

# Build a sequential model
# One input layer and two hidden layers, each hidden layer is followed by a Dropout layer to reduce overfitting.
# Each hidden layer uses the ReLU activation function, and the last output layer uses the Sigmoid activation function for binary classification.
# Use L2 regularizer (regularizer_l2) to reduce overfitting.
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', 
              kernel_regularizer = regularizer_l2(0.001), 
              input_shape = dim(x_train)[2]) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 16, activation = 'relu',
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile the model, set the Adam optimizer (learning rate 0.0005), the loss function is binary cross-entropy, and the performance measure is accuracy.
model %>% compile(
  optimizer = optimizer_adam(lr = 0.0005),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

model

# Use early stopping (callback_early_stopping) to prevent overfitting and end training early if the loss on the validation set does not improve after a certain number of iterations.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)

# Training model
history <- model %>% fit(
  x_train,
  y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(early_stop)
)

# Evaluate model performance and obtain loss values and accuracy.
model %>% evaluate(x_test, y_test)
# Draw training history objects to show the loss and accuracy changes during training and verification, helping to analyze the learning process of the model, such as whether it is overfitting or underfitting.
plot(history)
