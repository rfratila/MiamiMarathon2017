library(tidyverse)

d <- read_csv("~/github/MiamiMarathon2017/R_data.csv")
d$X1 <- NULL
d$Id <- NULL
d$Rank <- NULL
d$num <- as.factor(d$num)
d$Year <- as.factor(d$Year)
d$Sex <- as.factor(d$Sex)
d <- d %>% filter(Year!=2013) %>% filter(num!=347)
d$ageFactor <- cut(d$`Age Category`, breaks=c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), include.lowest=TRUE, right=FALSE)

