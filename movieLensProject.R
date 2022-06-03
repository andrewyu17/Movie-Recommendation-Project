################################################################################
## Movie Recommendation Model
## By Andrew Yu
## Last Updated: 6/3/22
################################################################################

##################################################
## 0 - Load libraries
##################################################
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(recosystem)
options(digits=4)


##################################################
## 1 - Creating Subset of MovieLens Dataset
##################################################
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##################################################
## 3 - Pre-Processing Data
##################################################

# Extract release year
edx$release <- as.numeric(str_extract(edx$title, "\\d{4}"))
validation$release <- as.numeric(str_extract(validation$title, "\\d{4}"))

# Drop some columns to save space
edx <- subset(edx, select = -c(timestamp))
validation <- subset(validation, select = -c(timestamp))

# Split into training and test
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

removed <- anti_join(temp, test)
train <- rbind(train, removed)
rm(test_index, temp, removed)

##################################################
## 5 - Modeling (Basic Effects and Regularization)
##################################################

############### Naive Model ###############
mu <- mean(train$rating)
mu

naive_rmse <- RMSE(test$rating, mu)
naive_rmse

results <- tibble(Model = "Just the average", RMSE = naive_rmse)
rm(naive_rmse)
results

############### Genre Effect Model ###############

genre_avgs <- train %>%  
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu))

qplot(b_g, data = genre_avgs, bins = 10, color = I("black"))

predicted_ratings <- mu + test %>% 
  left_join(genre_avgs, by='genres') %>%
  pull(b_g)

genre_rmse <- RMSE(predicted_ratings, test$rating)
results <- bind_rows(results, 
                     tibble(Model="Genre Effects Model", 
                            RMSE = genre_rmse))
rm(genre_rmse)
results

############### Year of Release Effect Model ###############

release_avgs <- train %>%
  group_by(release) %>%
  summarize(b_y = mean(rating - mu))

qplot(b_y, data = release_avgs, bins = 10, color = I("black"))

predicted_ratings <- mu + test %>% 
  left_join(release_avgs, by='release') %>%
  pull(b_y)

release_rmse <- RMSE(predicted_ratings, test$rating)
results <- bind_rows(results, 
                     tibble(Model="Release Year Effects Model", 
                            RMSE = release_rmse))
rm(release_rmse)
results

##### Movie Effects Model
movie_avgs <- train %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

predicted_ratings <- mu + test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_rmse <- RMSE(predicted_ratings, test$rating)
results <- bind_rows(results, 
                     tibble(Model="Movie Effects Model", 
                                RMSE = movie_rmse))
rm(movie_rmse)
results

############### User-Movie Effects Model ###############
user_avgs <- train %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

train %>% 
  group_by(userId) %>% 
  filter(n()>=100) %>%
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

userMovie_rmse <- RMSE(predicted_ratings, test$rating)
results <- bind_rows(results, 
                     tibble(Model="User-Movie Effects Model", 
                            RMSE = userMovie_rmse))
rm(userMovie_rmse)
results

############### Regularization ###############

# Check movie effects
test %>%
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%  
  slice(1:10) %>% 
  pull(title) 
# Most issues came with ratings of Shawshank

# Penalized Least Squares

# Find optimal lambda
set.seed(1)
kfolds <- createFolds(edx$rating, k=10)

lambdas <- seq(0, 7, 0.1)
rmses <- matrix(nrow=10,ncol=71)

for (k in 1:5){
  train_k <- edx[-kfolds[[k]],]
  temp <- edx[kfolds[[k]],]
  test_k <- temp %>% 
    semi_join(train_k, by = "movieId") %>%
    semi_join(train_k, by = "userId")
  
  removed <- anti_join(temp, test_k)
  train_k <- rbind(train_k, removed)
  rm(temp, removed)
  
  muk <- mean(train_k$rating)
  
  just_the_sum <- train_k %>% 
    group_by(movieId) %>% 
    summarize(s = sum(rating - muk), n_i = n())
  
  rmses[k,] <- sapply(lambdas, function(l){
    predicted_ratings <- test_k %>% 
      left_join(just_the_sum, by='movieId') %>% 
      mutate(b_i = s/(n_i+l)) %>%
      mutate(pred = muk + b_i) %>%
      pull(pred)
    return(RMSE(predicted_ratings, test_k$rating))
  })
}
rm(train_k, test_k, muk)

rmses_avg <- colMeans(rmses)
qplot(lambdas, rmses_avg)  
lambda <- lambdas[which.min(rmses_avg)]

# 2.1 is the optimal lambda

movie_reg_avgs <- train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

tibble(original = movie_avgs$b_i, 
       regularized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Test new model
predicted_ratings <- test %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
regMovie_rmse <- RMSE(predicted_ratings, test$rating)
results <- bind_rows(results, 
                     tibble(Model="Regularized Movie Effects Model", 
                            RMSE = regMovie_rmse))
rm(regMovie_rmse)
results
# Improves only by a tiny bit

############### Regularize User-Movie Effects Model ############### 

rmses <- matrix(nrow=10,ncol=71)

for (k in 1:10){
  train_k <- edx[-kfolds[[k]],]
  temp <- edx[kfolds[[k]],]
  test_k <- temp %>% 
    semi_join(train_k, by = "movieId") %>%
    semi_join(train_k, by = "userId")
  
  removed <- anti_join(temp, test_k)
  train_k <- rbind(train_k, removed)
  rm(temp, removed)
  
  muk <- mean(train_k$rating)
  
  just_the_sum <- train_k %>% 
    group_by(movieId) %>% 
    summarize(s = sum(rating - muk), n_i = n())
  
  rmses[k,] <- sapply(lambdas, function(l){
    
    b_i <- train_k %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - muk)/(n()+l))
    
    b_u <- train_k %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - muk)/(n()+l))
    
    predicted_ratings <- 
      test_k %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = muk + b_i + b_u) %>%
      pull(pred)
    
    return(RMSE(predicted_ratings, test_k$rating))
  })
}

rm(train_k, test_k, muk)

rmses_avg <- colMeans(rmses)
qplot(lambdas, rmses_avg)  
lambda <- lambdas[which.min(rmses_avg)] # 4.9 is optimal lambda

b_i <- train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- train %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda), n_i = n())

# Test new model
predicted_ratings <- test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
regUserMovie_rmse2 <- RMSE(predicted_ratings, test$rating)
results <- bind_rows(results, 
                     tibble(Model="Final Regularized User-Movie Effects Model", 
                            RMSE = regUserMovie_rmse2))
results
# Pretty much no change

##################################################
## 6 - Modeling using Matrix Factorization
##################################################


train_data <- data_memory(user_index = train$userId, 
                          item_index = train$movieId,
                          rating = train$rating, index1 = TRUE)
test_data <- data_memory(user_index = test$userId, 
                          item_index = test$movieId,
                          rating = test$rating, index1 = TRUE)
recommender <- Reco()

factors <- seq(10,30,10)
iterations <- seq(100,500,200)
args <- expand.grid(x = factors, y = iterations)

mf_tuning <- mapply(function(x,y){
  recommender$train(train_data, opts = c(dim = x, costp_l2 = 0.1, costq_l2 = 0.1, 
                                         lrate = 0.1, niter = y, nfold = 10, 
                                         nthread = 6, verbose = F))
  test$prediction <- recommender$predict(test_data, out_memory())
  return(c(f = x, i = y, mf_rmses = RMSE(test$prediction, test$rating)))
}, x = args$x, y = args$y)
mf_tuning <- as.data.frame(t(mf_tuning))
mf_tuning[which.min(mf_tuning$mf_rmses),]
mf_rmse <- which.min(mf_tuning$mf_rmses)
results <- bind_rows(results, 
                     tibble(Model="Matrix Factorization Model", 
                            RMSE = mf_rmse))
# 30 factors and 500 iterations produces the lowers RMSE at 0.8098

################################################## 
## 7 - Utilizing Best models on Edx and Validation
##################################################

############### Model 1 ###############
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+4.9))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+4.9), n_i = n())

# Test new model
predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model1 <- RMSE(predicted_ratings, validation$rating)

final_results <- tibble(Model= 'Regularized User-Movie Effects Model', 
                        RMSE = model1)

############### Model 2 ###############
edx_data <- data_memory(user_index = edx$userId, 
                          item_index = edx$movieId,
                          rating = edx$rating, index1 = TRUE)
validation_data <- data_memory(user_index = validation$userId, 
                         item_index = validation$movieId,
                         rating = validation$rating, index1 = TRUE)

recommender <- Reco()

recommender$train(edx_data, opts = c(dim = 30, costp_l2 = 0.1, costq_l2 = 0.1, 
                                       lrate = 0.1, niter = 500, nfold = 10, 
                                       nthread = 6, verbose = F))

validation$prediction <- recommender$predict(validation_data, out_memory())
model2 <- RMSE(validation$prediction, validation$rating)
final_results <- bind_rows(final_results, 
                           tibble(Model= 'Matrix Factorization Model', 
                                  RMSE = model2))







