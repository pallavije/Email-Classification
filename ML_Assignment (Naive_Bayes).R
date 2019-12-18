
# Install Required Packages

install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)
install.packages("wordcloud")
library('wordcloud')
library(readr)

# Load the dataset and check the Stucture of the data

emails_raw = read_csv("C:/Users/hp/Downloads/emails_1.csv")
str(emails_raw)

#convert to factor

emails_raw$spam <- factor(emails_raw$spam)
str(emails_raw$spam)
table(emails_raw$spam)


# Pre-processing

email_corpus <- VCorpus(VectorSource(emails_raw$text))
print(email_corpus)

#To receive a summary of specific email, we can use the inspect() function with list operators

inspect(email_corpus[1:2])

#to view the email

as.character(email_corpus[[2]])

lapply(email_corpus[1:2], as.character)


#The tm_map() function provides a method to apply a transformation . We will use 
#this function to clean up our corpus using a series of transformations
#1st convert all to lower

email_corpus_clean <- tm_map(email_corpus,
                             content_transformer(tolower))


# Removing stopwords, Numbers, punctuation and whitespaces

email_corpus_clean <- tm_map(email_corpus_clean,
                             removeWords, stopwords("english"))

email_corpus_clean <- tm_map(email_corpus_clean, removeWords,c("subject"))
email_corpus_clean <- tm_map(email_corpus_clean, removeNumbers)
email_corpus_clean <- tm_map(email_corpus_clean, removePunctuation)
email_corpus_clean <- tm_map(email_corpus_clean, stripWhitespace)

#check it worked

as.character(email_corpus[[100]])

as.character(email_corpus_clean[[100]])

# Standardization for text data involves reducing words to their root form in a process called stemming.

email_corpus_clean <- tm_map(email_corpus_clean, stemDocument)


#Data preparation - splitting text documents into words using Document Term Matrix (DTM)

email_dtm <- DocumentTermMatrix(email_corpus_clean)

#DTM cleaning Method 2

email_dtm2 <- DocumentTermMatrix(email_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

# Differences

email_dtm

email_dtm2


# Data preparation - creating training and test datasets (75% for train and 25% for test)

email_dtm_train <- email_dtm[1:4296, ]  
email_dtm_test <- email_dtm[4297:5728, ]

# Labels

email_train_labels <- emails_raw[1:4296, ]$spam
email_test_labels <- emails_raw[4297:5728, ]$spam

prop.table(table(email_train_labels))
prop.table(table(email_test_labels))




# Visualizing text data - word clouds

wordcloud(email_corpus_clean, min.freq = 1000, random.order = FALSE, col=rainbow(7))

# Subset where the email type is spam:

spam <- subset(emails_raw, spam == "1")
ham <- subset(emails_raw, spam == "0")  


wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))  

# Data preparation - creating indicator features for frequent words

# Words appearing aleast 5 time below

findFreqTerms(email_dtm_train,20)

email_freq_words <- findFreqTerms(email_dtm_train, 20)              
str(email_freq_words)

# Filter only appearing in certain vector

email_dtm_freq_train<- email_dtm_train[ , email_freq_words]
email_dtm_freq_test <- email_dtm_test[ , email_freq_words]

#The following defines a convert_counts() function to convert counts to Yes/No strings:

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

#We need to apply convert_counts() to each of the columns in our sparse

#MARGIN parameter to specify either rows or columns.  MARGIN = 2columns, MARGIN = 1 is rows

email_train <- apply(email_dtm_freq_train, MARGIN = 2,
                     convert_counts)

email_test <- apply(email_dtm_freq_test, MARGIN = 2,
                    convert_counts)



# The Naive Bayes implementation

install.packages("e1071")
library(e1071)
install.packages("caret")
library(caret)
library(gmodels)

email_classifier <- naiveBayes(email_train, email_train_labels)

# Evaluating model performance make the predictions and Cross Table Validation

predictions <- predict(email_classifier, email_test)

CrossTable(predictions, email_test_labels,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))

# Summarize Results (confusion Matrix and Accuracy)

confusionMatrix(predictions, email_test_labels)


# Improving model performance

email_classifier_1 <- naiveBayes(email_train, email_train_labels,
                                laplace = 1)

predictions_1 <- predict(email_classifier_1, email_test)

CrossTable(predictions_1, email_test_labels, 
           prop.chisq = FALSE, prop.t = FALSE, 
           prop.r = FALSE, dnn = c('predicted', 'actual'))

# Improved Model performace Summarize Results (confusion Matrix and Accuracy) 

confusionMatrix(predictions_1, email_test_labels)

# Realtime Test

test.data <- data.frame(Text="Hi Pallu, I sent the assignment file. Check it and let me know")
test_pred = predict(email_classifier_1, newdata=test.data, interval="prediction")

if (test_pred==0){
  print("The Email is Ham")
}else{
  print("The Email is Spam")
}
