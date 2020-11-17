# renv::init() # инициализация виртуального окружения
# renv::install("devtools") # установка библиотеки из CRAN
# renv::snapshot() # делаем снимок версий библиотек в нашем виртуальном окружении
# фиксируем этот список в .lock-файле для возможности восстановления
# renv::restore() # команда отктиться к предыдушему удачному обновления библиотек

# ------------------- 
# Лабораторная работа №13:
# Однослойные нейронные сети. Nnet, Neuralnet.


#  –аспознавание сорта оливкового масла

icsv <- read.csv("iris.csv",header = T, sep = ",")

for(i in 1:50){ #Setosa
  icsv[i,5]<-1
}
for(i in 51:100){#Versicolor

  icsv[i,5]<-2
}
for(i in 101:150){#Virginica
  icsv[i,5]<-3
}

a <- sapply(icsv[ , -5], min)
b <- sapply(icsv[ , -5], max) - a

#  —обственно стандартизаци€ входных переменных
icsv.x <- scale(icsv[, 1:4], center=a, scale=b)

#  Преобразование выходной переменной в три столбца, в три индикаторные переменные.
y1 <- rep(0, nrow(icsv))
y1[icsv[ , 5]==1] <-1

y2 <- rep(0, nrow(icsv))
y2[icsv[ , 5]==2] <-1

y3 <- rep(0, nrow(icsv))
y3[icsv[ , 5]==3] <-1

z.1 <- as.data.frame(cbind(icsv.x, y1, y2, y3))

# создание тестовой и обучающей выборки
set.seed(1234567)
index <- sample(1:nrow(z.1), round(nrow(z.1)*2/3), replace=F)
z.train <- z.1[index,]
z.test <- z.1[-index,]


#  Подключаем библиотеку neuralnet.
library(neuralnet)

num.nets <- 10

seed.start <- 12345

error.best <- 1

error.vector <- rep(-9999, num.nets)
seed.current <- seed.start

for (i in 1: num.nets){ 
  
  seed.current <- seed.current + 1
  set.seed(seed.current)
  nn.temp <- neuralnet( y1+y2+y3 ~  sepal.length + sepal.width + petal.length + petal.width,
                        data=z.train, hidden = c(3,2), linear.output=F)

  res.z <- compute(nn.temp, z.train[, 1:4] )  
  res.z2 <- apply(res.z$net.result, 1, which.max )
  error.temp <- sum(res.z2 != icsv[index,1] )/length(index)
  error.vector[i] <- error.temp
  if (error.temp < error.best)
  {
    nn.best <- nn.temp
    error.best <- error.temp
    seed.best <- seed.current
  }
}

plot(nn.best)

error.temp

error.vector

seed.best


# таблица сопряженности для лучшей i—
res.3 <- compute(nn.best, z.train[, 1:4] )    # обучающая выборка
res.z3 <- apply(res.3$net.result, 1, which.max)
table(res.z3, icsv[index,1])
#  res.z3   1   2   3
#  1   215      0   0
#  2    0      70   0
#  3   0        0   96
res.4 <- compute(nn.best, z.test[, 1:4] )     # тестовая выборка
res.z4 <- apply(res.4$net.result, 1, which.max )
table(res.z4, icsv[-index,1])
#  res.z4   1   2   3
#  1       108  0   0
#  2        0  28   0
#  3        0   0  55
sum(diag(table(icsv[-index,1], res.z4)))/length(icsv[-index,1])*100
#  [1] 97.38219895
100 - (sum(diag(table(icsv[-index,1], res.z4)))/length(icsv[-index,1])*100)
#  [1] 2.617801047