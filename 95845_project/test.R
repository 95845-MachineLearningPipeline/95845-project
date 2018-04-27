library(dplyr)
library(tidyr)
library(lubridate)
library(tcltk)

hist_consumption = read.csv("train.csv")
building_meta = read.csv("metadata.csv")
hist_weather = read.csv("weather.csv")
holiday = read.csv("date_info.csv")
submission_format = read.csv("submission_format.csv")

building_2 <- hist_consumption %>% filter(SiteId == 2)
building_2$Timestamp <- ymd_hms(building_2$Timestamp)

building2_weather <- hist_weather %>% filter(SiteId == 2)
building2_weather$Timestamp <- ymd_hms(building2_weather$Timestamp)

building2.data <- building_2 %>% select(-ForecastId, -obs_id) %>%
  left_join(building2_weather %>% select(-X, -SiteId), by = "Timestamp") %>%
  filter(!is.na(Temperature)) %>%
  group_by(SiteId, Timestamp, Value) %>%
  summarise(OutsideTemperature = mean(Temperature))

count = 0
# data <- safe copy of building2.data
# add ID for each row to building2.data to create data
data <- building2.data %>% filter(!is.na(Value))
data$ID <- seq.int(nrow(data))

# ID - 24, ID - 48, ID - 72, ID - 1, ID - 2, ID - 3, ID
days_3 <- 72
days_2 <- 48
days_1 <- 24
hours_3 <- 3
hours_2 <- 2
hours_1 <- 1
total <- 20000
# create a progress bar
pb <- tkProgressBar(title = "Data Cleaning Progress", min = 0,
                    max = total, width = 300)

preprocessData <- function(dat) {
  data <-data.frame(index = integer(),
                   Timestamp = as.POSIXct(factor()),
                   T_72 = double(),
                   T_48 = double(),
                   T_24 = double(),
                   T_3 = double(),
                   T_2 = double(),
                   T_1 = double(),
                   T_0 = double(),
                   Sun = integer(),
                   Mon = integer(),
                   Tue = integer(),
                   Wed = integer(),
                   Thu = integer(),
                   Fri = integer(),
                   Sat = integer(),
                   Closed = integer(),
                   OutsideTemperature = double(),
                   Output = double(),
                   ID = integer())
  for (i in 1:nrow(dat)) {
    if (i > days_3 && (i - days_3) <= total) {
      realIndex <- i - days_3
      data[realIndex, ]$index <- realIndex
      data[realIndex, ]$Timestamp <- dat[i, ]$Timestamp
      data[realIndex, ]$ID <- i
      data[realIndex, ]$Output <- dat[i + 1, ]$Value
      data[realIndex, ]$T_72 <- dat[i - days_3, ]$Value
      data[realIndex, ]$T_48 <- dat[i - days_2, ]$Value
      data[realIndex, ]$T_24 <- dat[i - days_1, ]$Value
      data[realIndex, ]$T_1 <- dat[i - hours_1, ]$Value
      data[realIndex, ]$T_2 <- dat[i - hours_2, ]$Value
      data[realIndex, ]$T_3 <- dat[i - hours_3, ]$Value
      data[realIndex, ]$T_0 <- dat[i, ]$Value
      data[realIndex, ]$OutsideTemperature <- dat[i, ]$OutsideTemperature
      
      if (wday(dat[i, ]$Timestamp) == 1) {
        data[realIndex, ]$Sun <- 1
        data[realIndex, ]$Closed <- 1
      } else if (wday(dat[i, ]$Timestamp) == 2) {
        data[realIndex, ]$Mon <- 0
        data[realIndex, ]$Closed <- 0
      } else if (wday(dat[i, ]$Timestamp) == 3) {
        data[realIndex, ]$Tue <- 1
        data[realIndex, ]$Closed <- 0
      } else if (wday(dat[i, ]$Timestamp) == 4) {
        data[realIndex, ]$Wed <- 1
        data[realIndex, ]$Closed <- 0
      } else if (wday(dat[i, ]$Timestamp) == 5) {
        data[realIndex, ]$Thu <- 1
        data[realIndex, ]$Closed <- 0
      } else if (wday(dat[i, ]$Timestamp) == 6) {
        data[realIndex, ]$Fri <- 1
        data[realIndex, ]$Closed <- 0
      } else if (wday(dat[i, ]$Timestamp) == 7) {
        data[realIndex, ]$Sat <- 1
        data[realIndex, ]$Closed <- 1
      }
      setTkProgressBar(pb, i, label=paste(round((i - days_3)/total*100, 0)), "done")
    }
  }
  data[is.na(data)] <- 0
  close(pb)
  return(data)
}

cleanedData <- preprocessData(data)
write.csv(cleanedData, "cleanedData.csv")

data.cleaned <- read.csv("cleanedData.csv") %>% select(-X)
data.cleaned$Timestamp <- as.POSIXct(data.cleaned$Timestamp)

# Basis function to do feature engineering
period = 24 #1 day
K = 5

sincos = function(dat, period=24, K=5) {
  data = dat
  for (i in 1:K) {
    data[[paste("sin_", i)]] = sin(hour(data$Timestamp)*i*2*pi/period)
  }
  for (i in 1:K) {
    data[[paste0("cos_", i)]] = cos(hour(data$Timestamp)*i*2*pi/period)
  }
  data
}

#data.basis <- sincos(data.cleaned)
data.basis <- read.csv("basis_function_results.csv") %>% rename(index = Timestamp)
data <- data.cleaned %>% right_join(data.basis, by="index") %>% filter(!is.na(T_0)) %>% filter(!is.na(Value))
# Split the data into train/test set
X <- data %>% select(-index, -Output, -ID, -Timestamp)
Y <- data %>% select(Output)
dims <- dim(data)
trainX <- X[1:(dims[1]), ] %>% as.matrix()
trainY <- Y[1:(dims[1]), ] %>% as.matrix()
testX <- X[(dims[1]/2):dims[1], ] %>% as.matrix()
testY <- Y[(dims[1]/2):dims[1], ] %>% as.matrix()

# Neural Net code to
library(keras)
model.NN <- keras_model_sequential()
hidden_size <- 27
model.NN %>%
  layer_dense(units = hidden_size, input_shape = c(ncol(trainX)), activation = 'relu',
              kernel_initializer='normal') %>%
  layer_activation_leaky_relu(alpha = 0.1) %>%
  layer_dense(units = 27, activation = 'relu', kernel_initializer = 'normal') %>%
  layer_dense(units = 54, activation = 'relu', kernel_initializer = 'normal') %>%
  layer_dense(units = 27, activation = 'relu', kernel_initializer = 'normal') %>%
  layer_dense(units = 1, activation = 'relu',  kernel_initializer = 'normal')

summary(model.NN)

# Train the NN model
model.NN %>% compile(
  loss = 'mse',
  optimizer = optimizer_nadam(clipnorm = 10),
  metrics = c('mse')
)

history = 
  model.NN %>% fit(trainX, trainY,
                   epochs = 100,
                   batch_size = 25,
                   shuffle=T
  )

model.c %>% evaluate(testX, testY)
model.c %>% get_weights()
plot(history)



