# Set working directory
setwd('C:\\Users\\amarl\\CDS501_RLab\\Assignment')
getwd()

# Import library
library(caret)
library(dplyr)
library(e1071)
library(rpart)
library(rattle)
library(ggpubr)
library(ggplot2)
library(reshape2)
library(tidyverse)
library(doParallel)
library(rpart.plot)
library(randomForest)

patientData = read_delim('public_health_behavior.csv')

#-----------------------------------------------------------------------------------------------------------------------------------------------
# DATA PREPARATION (Alia Marliana Binti Shaiful Bahari)
# Change the data type from numerical to categorical
patientData$Diabetes_012 = factor(patientData$Diabetes_012, levels = c(0, 1, 2), labels = c("No Diabetes", "Prediabetes", "Diabetes"))
patientData$HighBP = factor(patientData$HighBP, levels = c(0, 1), labels = c("No High BP", "High BP"))
patientData$HighChol = factor(patientData$HighChol, levels = c(0, 1), labels = c("No High Cholesterol", "High Cholesterol"))
patientData$HeartDisease = factor(patientData$HeartDisease, levels = c(0, 1), labels = c("No Heart Disease", "Heart Disease"))
patientData$Stroke = factor(patientData$Stroke, levels = c(0, 1), labels = c("No Stroke", "Stroke"))
patientData$Smoker = factor(patientData$Smoker, levels = c(0, 1), labels = c("Non-Smoker", "Smoker"))
patientData$FreqSmoke = factor(patientData$FreqSmoke, levels = c(0, 1, 2), labels = c("Everyday", "Some Days", "Not at All"))
patientData$FreqEcig = factor(patientData$FreqEcig, levels = c(0, 1, 2), labels = c("Everyday", "Some Days", "Not at All"))

patientData$PhysActivity = factor(patientData$PhysActivity, levels = c(0, 1), labels = c("No", "Yes"))
patientData$Fruits = factor(patientData$Fruits, levels = c(0, 1), labels = c("No", "Yes"))
patientData$Veggies = factor(patientData$Veggies, levels = c(0, 1), labels = c("No", "Yes"))
patientData$HvyAlcoholConsump = factor(patientData$HvyAlcoholConsump, levels = c(0, 1), labels = c("No", "Yes"))
patientData$GenHlth = factor(patientData$GenHlth, levels = 1:5, labels = c("Excellent", "Very Good", "Good", "Fair", "Poor"))
patientData$MentHlth = cut(patientData$MentHlth, breaks = c(-Inf, 11, 21, 31), labels = c("0-10", "11-20", "21-30"), right = FALSE)
patientData$PhysHlth = cut(patientData$PhysHlth, breaks = c(-Inf, 11, 21, 31), labels = c("0-10", "11-20", "21-30"), right = FALSE)
patientData$DiffWalk = factor(patientData$DiffWalk, levels = c(0, 1), labels = c("No", "Yes"))
patientData$Sex = factor(patientData$Sex, levels = c(0, 1), labels = c("Female", "Male"))
patientData$Age = factor(patientData$Age, levels = 1:13, labels = c("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"))
patientData$Education = factor(patientData$Education, levels = 1:6, labels = c("Never Attended", "Elementary", "Some High School", "High School Graduate", "College 1-3 Years", "College 4+ Years"))

patientData$Employment = factor(patientData$Employment, levels = 0:7, labels = c("Employed", "Self-Employed", "Out of Work (1 Year)", "Out of Work (More than 1 Year)", "Homemaker", "Student", "Retired", "Unable to Work"))
patientData$Income = factor(patientData$Income, levels = 1:11, labels = c("< $10,000", "< $15,000", "< $20,000", 
                                                                         "< $25,000", "< $35,000", "< $50,000", 
                                                                         "< $75,000", "< $100,000", "< $150,000", 
                                                                         "< $200,000", "$200,000 >="))
patientData$AnyHealthcare = factor(patientData$AnyHealthcare, levels = c(0, 1), labels = c("No", "Yes"))
patientData$NoDocbcCost = factor(patientData$NoDocbcCost, levels = c(0, 1), labels = c("No", "Yes"))
patientData$LastCheckup = factor(patientData$LastCheckup, levels = 1:6, labels = c("Within Past Year", "Within Past 2 Years", "Within Past 5 Years", "5 or More Years Ago", "Not Sure", "Never"))

# Check if there are any NA values in the entire data frame
missing_values = anyNA(patientData)
print(missing_values)

# FreqEcig have 14K NA's value
# Checking if FreqSmoke and FreqEcig is independent
cols = c('FreqSmoke', 'FreqEcig')
subset_data = patientData[cols]
c_table = table(subset_data)
print(c_table)
# The value for is p-value < 2.2e-16 showing that they are dependent to each other, so it is safe to remove the column FreqEcig 
# as both of them will give the same result 
chisq.test(c_table, correct=F)

# Remove the column that have missing values 
cols_remove = c('FreqEcig')
patientData = patientData %>% select(-cols_remove)

# To view the outliers in the BMI feature
boxplot(patientData$BMI, 
        main = "Boxplot of BMI", 
        ylab = "BMI", 
        col = "lightblue")

# To handle outlier in BMI column, need to change it to categorical.
# Source for classification of BMI level https://www.cdc.gov/obesity/basics/adult-defining.html
# Underweight: BMI < 18.5
# Healthy Weight: 18.5 ≤ BMI < 25
# Overweight: 25 ≤ BMI < 30
# Obesity: BMI ≥ 30, with subcategories:
# Class 1: 30 ≤ BMI < 35
# Class 2: 35 ≤ BMI < 40
# Class 3: BMI ≥ 40 (severe obesity)
# Define the breaks and corresponding labels for BMI categories
breaks = c(-Inf, 18.5, 25, 30, 35, 40, Inf)  # Define breakpoints for categories
labels = c("Underweight", "Healthy Weight", "Overweight", "Obesity Class 1", 
            "Obesity Class 2", "Obesity Class 3")  # Define category labels
patientData$BMI = cut(patientData$BMI, breaks = breaks, labels = labels, right = FALSE)

# To view all summary data and to show (other) value
summary(patientData, maxsum = max(lengths(lapply(patientData, unique))))

# FOR target = Heart Disease

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Exploratory Data Analysis 1 
# (Do socio-demographic factors influence the health status of an individual? - Joyce Lim Xinjie)  
# Scale the data according to the need
df2 <- patientData[, !names(patientData) %in% c("HighBP", "HighChol", "Smoker", "FreqSmoke", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "BMI")]

#scale data with only heart disease patients
new_df <- subset(df2, HeartDisease == 'Heart Disease') 

#barplot for age
ageplot <- ggplot(data=new_df, aes(x=factor(Age))) +
  geom_bar(stat="count", fill="steelblue") +
  ggtitle("Relationship between Age and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Age") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))

#barplot for gender
sexplot <- ggplot(data=new_df, aes(x=factor(Sex), fill = Sex)) +
  geom_bar(stat="count") +
  ggtitle("Relationship between Gender and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Gender") + 
  scale_fill_manual(values=c("pink", "lightblue"))

#barplot for education
educationplot <- ggplot(data=new_df, aes(x=factor(Education))) +
  geom_bar(stat="count", fill="steelblue") +
  ggtitle("Relationship between Education Level and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Education Level") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))

#barplot for income
incomeplot <- ggplot(data=new_df, aes(x=factor(Income))) +
  geom_bar(stat="count", fill="steelblue") +
  ggtitle("Relationship between Income Level and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Income Level") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))

#barplot for employment
employmentplot <- ggplot(data=new_df, aes(x=factor(Employment))) +
  geom_bar(stat="count", fill="steelblue") +
  ggtitle("Relationship between Employment Status and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Employment Status") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))

#barplot for cost
costplot <- ggplot(data=new_df, aes(x=factor(NoDocbcCost))) +
  geom_bar(stat="count", fill="steelblue") +
  ggtitle("Relationship between Economic Issue and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Unable to See a Doctor because of Cost in 1 Year") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))

#barplot for checkup
checkupplot <- ggplot(data=new_df, aes(x=factor(LastCheckup))) +
  geom_bar(stat="count", fill="steelblue") +
  ggtitle("Relationship between Frequency of Medical Checkup and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Last Medical Checkup") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))

#combine the plots into an image
ggarrange(ageplot, sexplot, educationplot, incomeplot, employmentplot, costplot, checkupplot, ncol = 2, nrow = 4)

# Perform correlation analysis 
df2$HeartDisease <- as.numeric(df2$HeartDisease)
df2$Sex <- as.numeric(df2$Sex)
df2$Age <- as.numeric(df2$Age)
df2$Education <- as.numeric(df2$Education)
df2$Income <- as.numeric(df2$Income)
df2$Employment <- as.numeric(df2$Employment)
df2$NoDocbcCost <- as.numeric(df2$NoDocbcCost)
df2$LastCheckup <- as.numeric(df2$LastCheckup)

# Compute correlation matrix
correlation_matrix = cor(df2[, c("HeartDisease", "Sex", "Age", "Education", "Income", "Employment", "NoDocbcCost", "LastCheckup")])
print(correlation_matrix)

# Heatmap
heatmap <- ggplot(data = melt(correlation_matrix), aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 1, size = 10, hjust = 1)) +
  labs(title = "Individual Socio-demographic Factors and Heart Disease", x = "", y = "") +
  coord_fixed()

# Adding correlation coefficients
heatmap + geom_text(aes(label = round(value, 2)), color = "darkgray", size = 4, vjust = 1)

#==================================================================================================
# Exploratory Data Analysis 2 (Florence)
# (Are lifestyle factors linked to the presence of heart disease? - Florence Tan Hui Ping)  

# Remove unrelated columns
colnames(patientData)
df3 = patientData[, !names(patientData) %in% c("HighBP", "HighChol", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Employment", "Income", "NoDocbcCost", "LastCheckup")]
colnames(df3)

# Analysing only the subset of patients with confirmed heart disease
new_df3 = subset(df3, HeartDisease == "Heart Disease")
new_df3

# Bar plot for BMI
BMIplot = ggplot(data=new_df3, aes(x=factor(BMI))) +
  geom_bar(stat="count", fill="darkgreen") +
  ggtitle("BMI vs. Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("BMI") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
BMIplot

# Bar plot for Smoking
smokeplot = ggplot(data=new_df3, aes(x=factor(FreqSmoke))) +
  geom_bar(stat="count", fill="darkgreen") +
  ggtitle("Smoking Frequency vs. Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Smoking Frequency") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
smokeplot

# Bar plot for Physical activity
physplot = ggplot(data=new_df3, aes(x=factor(PhysActivity))) +
  geom_bar(stat="count", fill="darkgreen") +
  ggtitle("Physical Activity vs. Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Physical Activity") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
physplot

# Bar plot for Fruits
fruitplot = ggplot(data=new_df3, aes(x=factor(Fruits))) +
  geom_bar(stat="count", fill="darkgreen") +
  ggtitle("Fruits Consumption vs. Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Fruits Consumption") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
fruitplot

# Bar plot for Veggies
vegeplot = ggplot(data=new_df3, aes(x=factor(Veggies))) +
  geom_bar(stat="count", fill="darkgreen") +
  ggtitle("Veggies Consumption vs. Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Veggies Consumption") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
vegeplot

# Bar plot for Alcohol Consumption
alcoholplot = ggplot(data=new_df3, aes(x=factor(FreqSmoke))) +
  geom_bar(stat="count", fill="darkgreen") +
  ggtitle("Heavy Alcohol vs. Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Heavy Drinker") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
alcoholplot

# Combine all lifestyle factor plots into an Image
ggarrange(BMIplot, smokeplot, physplot, fruitplot, vegeplot, alcoholplot, ncol = 2, nrow = 3)

# Perform correlation analysis to see the strength/direction of relationship between factors and Heart Disease
# Change attributes to numerical values
df3$BMI = as.numeric(factor(df3$BMI, levels = c("Underweight", "Healthy Weight", "Overweight", "Obesity Class 1", "Obesity Class 2", "Obesity Class 3")))
df3$HeartDisease = as.numeric(df3$HeartDisease, levels = c(0, 1), labels = c("No Heart Disease", "Heart Disease"))
df3$FreqSmoke = as.numeric(df3$FreqSmoke, levels = c(0, 1, 2), labels = c("Everyday", "Some Days", "Not at All"))
df3$PhysActivity = as.numeric(df3$PhysActivity, levels = c(0, 1), labels = c("No", "Yes"))
df3$Fruits = as.numeric(df3$Fruits, levels = c(0, 1), labels = c("No", "Yes"))
df3$Veggies = as.numeric(df3$Veggies, levels = c(0, 1), labels = c("No", "Yes"))
df3$HvyAlcoholConsump = as.numeric(df3$HvyAlcoholConsump, levels = c(0, 1), labels = c("No", "Yes"))

# Get correlation matrix
correlation_matrix = cor(df3[, c("HeartDisease", "FreqSmoke", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "BMI")])

# Extract heatmap of correlation values
heatmap_df3 = ggplot(data = reshape2::melt(correlation_matrix), aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 1, size = 10, hjust = 1)) +
  labs(title = "Individual Lifestyle Factors and Heart Disease", x = "", y = "") +
    coord_fixed()

# Add in correlation coefficients
heatmap_df3 + geom_text(aes(label = round(value, 2)), color = "darkgray", size = 4, vjust = 1)

#==============================================================================================================================
# Exploratory Data Analysis 3 
# (How do individual health indicators collectively contribute to the prevalence of heart disease? - Chew Yee Ling)

# Columns to include 
included_columns <- c("GenHlth","MentHlth","PhysHlth","DiffWalk","HighBP","HighChol","HeartDisease")
df4 <- patientData[,included_columns]

new_df4 <- subset(df4, HeartDisease == 'Heart Disease')

# Bar plot for General health
generalhealthplot = ggplot(data=new_df4, aes(x=factor(GenHlth))) +
  geom_bar(stat="count", fill="violet") +
  ggtitle("General Health and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("General Health") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
generalhealthplot

# Bar plot for Mental health
mentalhealthplot = ggplot(data=new_df4, aes(x=factor(MentHlth))) +
  geom_bar(stat="count", fill="violet") +
  ggtitle("Poor Mental Health and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("No. of Days of Poor Mental Health") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
mentalhealthplot

# Bar plot for Physical health
physicalhealthplot = ggplot(data=new_df4, aes(x=factor(PhysHlth))) +
  geom_bar(stat="count", fill="violet") +
  ggtitle("Poor Physical Health and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("No. of Days of Poor Physical Health") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
physicalhealthplot

# Bar plot for Mobility limitation
mobilitylimitationplot = ggplot(data=new_df4, aes(x=factor(DiffWalk))) +
  geom_bar(stat="count", fill="violet") +
  ggtitle("Mobility Limitation and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("Mobility Limitation") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
mobilitylimitationplot

# Bar plot for High BP
highbpplot = ggplot(data=new_df4, aes(x=factor(HighBP))) +
  geom_bar(stat="count", fill="violet") +
  ggtitle("High BP and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("High BP") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
highbpplot

# Bar plot for High Cholesterol
highcholplot = ggplot(data=new_df4, aes(x=factor(HighChol))) +
  geom_bar(stat="count", fill="violet") +
  ggtitle("High Cholesterol and Heart Disease") +
  ylab("No. of Diagnosed Heart Disease") + xlab("High Cholesterol") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))
highcholplot

# Combine all individual health indicators plots into an image
ggarrange(generalhealthplot, mentalhealthplot, physicalhealthplot, mobilitylimitationplot, highbpplot, highcholplot, ncol = 3, nrow = 2)

# Perform correlation analysis 
# Change attributes to numerical values
df4$HeartDisease <- as.numeric(df4$HeartDisease)
df4$GenHlth <- as.numeric(df4$GenHlth)
df4$MentHlth <- as.numeric(df4$MentHlth)
df4$PhysHlth <- as.numeric(df4$PhysHlth)
df4$DiffWalk <- as.numeric(df4$DiffWalk)
df4$HighBP <- as.numeric(df4$HighBP)
df4$HighChol <- as.numeric(df4$HighChol)

# Compute correlation matrix
correlation_matrix = cor(df4[, c("HeartDisease", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "HighBP", "HighChol")])
print(correlation_matrix)

# Heatmap
heatmap <- ggplot(data = melt(correlation_matrix), aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +
  theme_minimal() +
  labs(title = "Individual Health Indicators and Heart Disease", x = "", y = "") +
  coord_fixed()

# Adding correlation coefficients
heatmap + geom_text(aes(label = round(value, 2)), color = "darkgray", size = 4, vjust = 1)

#-----------------------------------------------------------------------------------------------------
# ASSIGNMENT 2
# FEATURE SELECTION
# Rearrange the column so that the target is in the first column
patientData <- patientData %>%
  select(HeartDisease, everything())

# Perform chi-square for feature selection
chi_results <- lapply(patientData[, -1], function(x) {
  chisq.test(table(patientData$HeartDisease, x))
})

# Extract statistics from chi-square results
chi_stats <- data.frame(
  Feature = names(patientData)[-1],
  Statistic = sapply(chi_results, function(result) result$statistic),
  P.Value = sapply(chi_results, function(result) result$p.value),
  DF = sapply(chi_results, function(result) result$parameter)
)

# Sort the data frame by p-value
chi_stats <- chi_stats[order(chi_stats$P.Value), ]

# Print the chi-square results table
print(chi_stats)

# Extract the names of the top 10 most significant features
top_10_features <- head(chi_stats$Feature, 10)

# Print the top 10 most significant features
print(top_10_features)

# Create new dataframe with target variable and top significant features
newPatientData <- patientData[, c("HeartDisease", top_10_features)]

# Print the new dataframe
print(head(newPatientData))

#---------------------------------------------------------------------------------------
# MODEL DEVELOPMENT

# Setting seed for reproducibility
set.seed(123)

# Generate random numbers for splitting
newPatientData$rgroup <- runif(nrow(newPatientData))

# Split into train, validation, and test sets
train_indices <- newPatientData$rgroup <= 0.6  # 60% for train
validation_indices <- newPatientData$rgroup > 0.6 & newPatientData$rgroup <= 0.8  # 20% for validation
test_indices <- newPatientData$rgroup > 0.8  # 20% for test

# Create train, validation, and test datasets
train_data <- newPatientData[train_indices, ]
validation_data <- newPatientData[validation_indices, ]
test_data <- newPatientData[test_indices, ]

# Remove the 'rgroup' column
train_data <- train_data[, !names(train_data) %in% "rgroup"]
validation_data <- validation_data[, !names(validation_data) %in% "rgroup"]
test_data <- test_data[, !names(test_data) %in% "rgroup"]

# Print the dimensions of train, validation, and test datasets
cat("Train dataset:", dim(train_data), "\n")
cat("Validation dataset:", dim(validation_data), "\n")
cat("Test dataset:", dim(test_data), "\n")

#========================================================================================
# MODEL 1: LOGISTIC REGRESSION
# Define train control with repeated cross-validation
ctrl <- trainControl(method = "repeatedcv",  # Repeated cross-validation
                     number = 10,            # Number of folds
                     repeats = 5)            # Number of repeats


model <- train(HeartDisease ~ .,        # Formula: HeartDisease as target variable
               data = train_data,      # Training data
               method = "glm",         # Logistic regression
               trControl = ctrl,       # Use defined train control
               tuneLength = 5)    

# Print the trained model
print(model)

# Print the best hyperparameter
print(model$bestTune) #result: none

# Make predictions on the validation set
predictions <- predict(model, newdata = validation_data)

# Generate the confusion matrix with specified positive class
cm <- confusionMatrix(predictions, validation_data$HeartDisease, mode = "everything", positive = "Heart Disease")

# Print the confusion matrix
print(cm)

# Extract metrics for each class
precision_yes <- cm$byClass["Pos Pred Value"]
recall_yes <- cm$byClass["Sensitivity"]
f1_yes <- 2 * (precision_yes * recall_yes) / (precision_yes + recall_yes)

# Calculate precision, recall, and F1 for the negative class
precision_no <- cm$byClass["Neg Pred Value"]
recall_no <- cm$byClass["Specificity"]
f1_no <- 2 * (precision_no * recall_no) / (precision_no + recall_no)

# Calculate class proportions
total <- sum(cm$table)
weight_no <- sum(cm$table[,"No Heart Disease"]) / total
weight_yes <- sum(cm$table[,"Heart Disease"]) / total

# Calculate weighted F1 score
weighted_f1 <- (f1_no * weight_no) + (f1_yes * weight_yes)

cat("Weighted F1 Score(Validation) for Logistic Regression:", weighted_f1, "\n")

#_____________________________________________________________________________________________________________________________
# MODEL EVALUATION (Logistic Regression)
# Make predictions on the test set
test_predictions <- predict(model, newdata = test_data)

# Generate the confusion matrix with specified positive class for test set
cm_test <- confusionMatrix(test_predictions, test_data$HeartDisease, mode = "everything", positive = "Heart Disease")

# Print the confusion matrix for test set
print(cm_test)

# Extract metrics for the positive class (Heart Disease)
precision_yes_test <- cm_test$byClass["Pos Pred Value"]
recall_yes_test <- cm_test$byClass["Sensitivity"]
f1_yes_test <- 2 * (precision_yes_test * recall_yes_test) / (precision_yes_test + recall_yes_test)

# Extract metrics for the negative class (No Heart Disease)
precision_no_test <- cm_test$byClass["Neg Pred Value"]
recall_no_test <- cm_test$byClass["Specificity"]
f1_no_test <- 2 * (precision_no_test * recall_no_test) / (precision_no_test + recall_no_test)

# Precision for Positive Class: 0.551282
# Precision for Negative Class: 0.919363
# Recall for Positive Class: 0.026993
# Recall for Negative Class: 0.998023

# Calculate class proportions
total_test <- sum(cm_test$table)
weight_no_test <- sum(cm_test$table[,"No Heart Disease"]) / total_test
weight_yes_test <- sum(cm_test$table[,"Heart Disease"]) / total_test

# Calculate weighted F1 score for test set
weighted_f1_test <- (f1_no_test * weight_no_test) + (f1_yes_test * weight_yes_test)

cat("Weighted F1 Score(Test) for Logistic Regression:", weighted_f1_test, "\n")

#============================================================================================
# MODEL 2: RANDOM FOREST
# Default hyperparameters
# Set up parallel processing
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Define train control with repeated cross-validation
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 5)

# Define a minimal tuning grid
minimal_rf_grid <- expand.grid(mtry = 4)  # Default mtry setting
ntree_initial <- 500  # Default number of trees

# Train the Random Forest model using default settings
default_rf_model <- train(
  HeartDisease ~ ., 
  data = train_data, 
  method = "rf", 
  trControl = ctrl,
  tuneGrid = minimal_rf_grid,  # Minimal tuning grid
  ntree = ntree_initial  # Default number of trees
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# Evaluate the model
print(default_rf_model)

# Make predictions on the validation set
predictions_rf <- predict(default_rf_model, newdata = validation_data)

# Generate the confusion matrix for the validation set
cm_rf <- confusionMatrix(predictions_rf, validation_data$HeartDisease, mode = "everything", positive = "Heart Disease")
print(cm_rf)

# Extract metrics for each class
precision_yes_rf <- cm_rf$byClass["Pos Pred Value"]
recall_yes_rf <- cm_rf$byClass["Sensitivity"]
f1_yes_rf <- 2 * (precision_yes_rf * recall_yes_rf) / (precision_yes_rf + recall_yes_rf)

precision_no_rf <- cm_rf$byClass["Neg Pred Value"]
recall_no_rf <- cm_rf$byClass["Specificity"]
f1_no_rf <- 2 * (precision_no_rf * recall_no_rf) / (precision_no_rf + recall_no_rf)

# Calculate class proportions
total_rf <- sum(cm_rf$table)
weight_no_rf <- sum(cm_rf$table[,"No Heart Disease"]) / total_rf
weight_yes_rf <- sum(cm_rf$table[,"Heart Disease"]) / total_rf

# Calculate weighted F1 score
weighted_f1_rf <- (f1_no_rf * weight_no_rf) + (f1_yes_rf * weight_yes_rf)
cat("Weighted F1 Score(Validation) for Random Forest:", weighted_f1_rf, "\n")

# Hyperparameter tuning of Random Forest algorithm
# Set up parallel processing
num_cores <- detectCores() - 1  # Use all but one core
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Set up repeated cross-validation with 10 folds and 5 repeats
set.seed(123)
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 5)

# Best hyperparameters
rf_grid <- expand.grid(mtry = 6)  
ntree <- 500  

# Train the Random Forest model using BEST hyperparameters
rf_model <- train(
  HeartDisease ~ ., 
  data = train_data, 
  method = "rf", 
  trControl = ctrl,
  tuneGrid = rf_grid,  
  ntree = ntree
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# Evaluate the model
print(rf_model)

# Make predictions on the validation set
predictions_rf <- predict(rf_model, newdata = validation_data)

# Generate the confusion matrix for the validation set
cm_rf <- confusionMatrix(predictions_rf, validation_data$HeartDisease, mode = "everything", positive = "Heart Disease")
print(cm_rf)

# Extract metrics for each class
precision_yes_rf <- cm_rf$byClass["Pos Pred Value"]
recall_yes_rf <- cm_rf$byClass["Sensitivity"]
f1_yes_rf <- 2 * (precision_yes_rf * recall_yes_rf) / (precision_yes_rf + recall_yes_rf)

precision_no_rf <- cm_rf$byClass["Neg Pred Value"]
recall_no_rf <- cm_rf$byClass["Specificity"]
f1_no_rf <- 2 * (precision_no_rf * recall_no_rf) / (precision_no_rf + recall_no_rf)

# Calculate class proportions
total_rf <- sum(cm_rf$table)
weight_no_rf <- sum(cm_rf$table[,"No Heart Disease"]) / total_rf
weight_yes_rf <- sum(cm_rf$table[,"Heart Disease"]) / total_rf

# Calculate weighted F1 score
weighted_f1_rf <- (f1_no_rf * weight_no_rf) + (f1_yes_rf * weight_yes_rf)
cat("Weighted F1 Score(Validation) for Random Forest:", weighted_f1_rf, "\n")

#_____________________________________________________________________________________________________________________________________
# RANDOM FOREST MODEL EVALUATION
# Make predictions on the test set
test_predictions_rf <- predict(rf_model, newdata = test_data)

# Generate the confusion matrix for the test set
cm_test_rf <- confusionMatrix(test_predictions_rf, test_data$HeartDisease, mode = "everything", positive = "Heart Disease")
print(cm_test_rf)

# Extract metrics for each class
precision_yes_test_rf <- cm_test_rf$byClass["Pos Pred Value"]
recall_yes_test_rf <- cm_test_rf$byClass["Sensitivity"]
f1_yes_test_rf <- 2 * (precision_yes_test_rf * recall_yes_test_rf) / (precision_yes_test_rf + recall_yes_test_rf)

precision_no_test_rf <- cm_test_rf$byClass["Neg Pred Value"]
recall_no_test_rf <- cm_test_rf$byClass["Specificity"]
f1_no_test_rf <- 2 * (precision_no_test_rf * recall_no_test_rf) / (precision_no_test_rf + recall_no_test_rf)

# Calculate class proportions
total_test_rf <- sum(cm_test_rf$table)
weight_no_test_rf <- sum(cm_test_rf$table[,"No Heart Disease"]) / total_test_rf
weight_yes_test_rf <- sum(cm_test_rf$table[,"Heart Disease"]) / total_test_rf

# Calculate weighted F1 score for the test set
weighted_f1_test_rf <- (f1_no_test_rf * weight_no_test_rf) + (f1_yes_test_rf * weight_yes_test_rf)
cat("Weighted F1 Score(Test) for Random Forest:", weighted_f1_test_rf, "\n")

#=======================================================================================================================
# MODEL 3: DECISION TREE
# Define train control with repeated cross-validation
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# Train the decision tree model with automatic hyperparameter tuning
model_dt <- train(HeartDisease ~ ., 
                  data = train_data, 
                  method = "rpart",
                  trControl = ctrl,
                  tuneLength = 10)  # Automatically determines the best hyperparameters

# Print the best hyperparameters
print(model_dt$bestTune)

# Extract the best model hyperparameters
best_hyperparameters <- model_dt$bestTune

# Retrain the model using the best hyperparameters (if desired)
final_model_dt <- rpart(HeartDisease ~ ., 
                        data = train_data, 
                        control = rpart.control(cp = best_hyperparameters$cp))

# Save the decision tree plot with the best hyperparameters
png(filename = "decision_tree_plot_best.png", width = 1600, height = 1200, res = 300)
rpart.plot(final_model_dt, type = 4, extra = 101)
dev.off()

# Make predictions on the validation set
validation_predictions <- predict(final_model_dt, newdata = validation_data, type = "class")

# Generate confusion matrix with specified positive class
cm <- confusionMatrix(validation_predictions, validation_data$HeartDisease, mode = "everything", positive = "Heart Disease")
print(cm)

# Extract and print metrics
precision_yes <- cm$byClass["Pos Pred Value"]
recall_yes <- cm$byClass["Sensitivity"]
f1_yes <- 2 * (precision_yes * recall_yes) / (precision_yes + recall_yes)
precision_no <- cm$byClass["Neg Pred Value"]
recall_no <- cm$byClass["Specificity"]
f1_no <- 2 * (precision_no * recall_no) / (precision_no + recall_no)

# Calculate class proportions
count_no_hd <- sum(cm$table[,"No Heart Disease"])
count_hd <- sum(cm$table[,"Heart Disease"])
total_count <- count_no_hd + count_hd

weight_no_hd <- count_no_hd / total_count
weight_hd <- count_hd / total_count

# Calculate weighted F1 score
weighted_f1 <- (f1_no * weight_no_hd) + (f1_yes * weight_hd)
weighted_f1_percent <- weighted_f1 * 100  # Convert to percentage
cat("Weighted F1 Score (Validation) for Decision Tree:", round(weighted_f1_percent, 2), "%\n")

#_________________________________________________________________________________________________________________________________
#MODEL EVALUATION (Decision Tree)
# Make predictions on the test set
test_predictions <- predict(final_model_dt, newdata = test_data, type = "class")

# Generate confusion matrix with specified positive class for test set
cm_test <- confusionMatrix(test_predictions, test_data$HeartDisease, mode = "everything", positive = "Heart Disease")
print(cm_test)

# Extract and print metrics for test set
precision_yes_test <- cm_test$byClass["Pos Pred Value"]
recall_yes_test <- cm_test$byClass["Sensitivity"]
f1_yes_test <- 2 * (precision_yes_test * recall_yes_test) / (precision_yes_test + recall_yes_test)
precision_no_test <- cm_test$byClass["Neg Pred Value"]
recall_no_test <- cm_test$byClass["Specificity"]
f1_no_test <- 2 * (precision_no_test * recall_no_test) / (precision_no_test + recall_no_test)

# Calculate class proportions for test set
count_no_hd_test <- sum(cm_test$table[,"No Heart Disease"])
count_hd_test <- sum(cm_test$table[,"Heart Disease"])
total_count_test <- count_no_hd_test + count_hd_test

weight_no_hd_test <- count_no_hd_test / total_count_test
weight_hd_test <- count_hd_test / total_count_test

# Calculate weighted F1 score for test set
weighted_f1_test <- (f1_no_test * weight_no_hd_test) + (f1_yes_test * weight_hd_test)
weighted_f1_test_percent <- weighted_f1_test * 100  # Convert to percentage

cat("Weighted F1 Score (Test) for Decision Tree:", round(weighted_f1_test_percent, 2), "%\n")

#=====================================================================================================
#MODEL 4: NAIVE BAYES
model_nb <- tune(naiveBayes, HeartDisease ~ ., data = train_data, tunecontrol = tune.control(sampling='cross', cross=10, nrepeat=5))
model_nb
best.model_nb <- model_nb$best.model

pred_nb_val <- predict(best.model_nb, newdata = validation_data)

# Generate the confusion matrix with specified positive class
cm <- confusionMatrix(pred_nb_val, validation_data$HeartDisease, mode = "everything", positive = "Heart Disease")
print(cm)

# Extract metrics for each class
precision_yes <- cm$byClass["Pos Pred Value"]
recall_yes <- cm$byClass["Sensitivity"]
f1_yes <- 2 * (precision_yes * recall_yes) / (precision_yes + recall_yes)

# Calculate precision, recall, and F1 for the negative class
precision_no <- cm$byClass["Neg Pred Value"]
recall_no <- cm$byClass["Specificity"]
f1_no <- 2 * (precision_no * recall_no) / (precision_no + recall_no)

# Calculate class proportions
total <- sum(cm$table)
weight_no <- sum(cm$table[,"No Heart Disease"]) / total
weight_yes <- sum(cm$table[,"Heart Disease"]) / total

# Calculate weighted F1 score
weighted_f1 <- (f1_no * weight_no) + (f1_yes * weight_yes)

cat("Weighted F1 Score(Validation) for Naive Bayes:", weighted_f1, "\n")

#_____________________________________________________________________________________
#MODEL EVALUATION (NAIVE BAYES)
pred_nb_test <- predict(best.model_nb, newdata = test_data)
# Generate the confusion matrix with specified positive class for test set
cm_test <- confusionMatrix(pred_nb_test, test_data$HeartDisease, mode = "everything", positive = "Heart Disease")

# Print the confusion matrix for test set
print(cm_test)

# Extract metrics for the positive class (Heart Disease)
precision_yes_test <- cm_test$byClass["Pos Pred Value"]
recall_yes_test <- cm_test$byClass["Sensitivity"]
f1_yes_test <- 2 * (precision_yes_test * recall_yes_test) / (precision_yes_test + recall_yes_test)

# Extract metrics for the negative class (No Heart Disease)
precision_no_test <- cm_test$byClass["Neg Pred Value"]
recall_no_test <- cm_test$byClass["Specificity"]
f1_no_test <- 2 * (precision_no_test * recall_no_test) / (precision_no_test + recall_no_test)

# Precision for Positive Class: 0.265740
# Precision for Negative Class: 0.944049
# Recall for Positive Class: 0.408035
# Recall for Negative Class: 0.898571
sprintf("Precision for Positive Class: %f", precision_yes_test)
sprintf("Precision for Negative Class: %f", precision_no_test)
sprintf("Recall for Positive Class: %f", recall_yes_test)
sprintf("Recall for Negative Class: %f", recall_no_test)

# Calculate class proportions
total_test <- sum(cm_test$table)
weight_no_test <- sum(cm_test$table[,"No Heart Disease"]) / total_test
weight_yes_test <- sum(cm_test$table[,"Heart Disease"]) / total_test

# Calculate weighted F1 score for test set
weighted_f1_test <- (f1_no_test * weight_no_test) + (f1_yes_test * weight_yes_test)

cat("Weighted F1 Score(Test) for Naive Bayes:", weighted_f1_test, "\n")

