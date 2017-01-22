library(tidyverse)

d <- read_csv("~/github/MiamiMarathon2017/R_data.csv")

d$X1 <- NULL
d$Rank <- NULL
d$Pace <- NULL
d$Name <- NULL
d$num <- as.factor(d$num)
d$Sex <- as.factor(d$Sex)
d$Year <- as.factor(d$Year)

d <- d %>%
	filter(Year!=2013) %>% 
	filter(num!=3d47)
levels(d$num) <- c("1","2","3","4","5","6","7",">7",">7",">7",">7",">7",">7",">7")

d$ageFactor <- cut(d$`Age Category`, breaks=c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), include.lowest=TRUE, right=FALSE)
d[d$ageFactor == "[0,10)",]$ageFactor <- NA

d$day_no <- rep(NA, length(d$Sex))
d$day_no[d$Year %in% c('2012', '2006')] <- 29
d$day_no[d$Year %in% c('2011', '2005')] <- 30 
d$day_no[d$Year %in% c('2010')] <- 31
d$day_no[d$Year %in% c('2009', '2015')] <- 25
d$day_no[d$Year %in% c('2008', '2013')] <- 27
d$day_no[d$Year %in% c('2007')] <- 28
d$day_no[d$Year %in% c('2003', '2014')] <- 33
d$day_no[d$Year %in% c('2004')] <- 32
d$day_no[d$Year %in% c('2016')] <- 24


d$temp <- rep(NA, length(d$Sex))
d$temp[d$Year=='2003'] <- 62
d$temp[d$Year=='2004'] <- 74
d$temp[d$Year=='2005'] <- 70
d$temp[d$Year=='2006'] <- 73
d$temp[d$Year=='2007'] <- 69
d$temp[d$Year=='2008'] <- 68
d$temp[d$Year=='2009'] <- 68
d$temp[d$Year=='2010'] <- 71
d$temp[d$Year=='2011'] <- 63
d$temp[d$Year=='2012'] <- 72
d$temp[d$Year=='2013'] <- 73
d$temp[d$Year=='2014'] <- 76
d$temp[d$Year=='2015'] <- 60
d$temp[d$Year=='2016'] <- 54

d$flu <- rep(NA, length(d$Sex))
d$flu[d$Year=='2005'] <- 2.37
d$flu[d$Year=='2006'] <- 2.76
d$flu[d$Year=='2007'] <- 1.2
d$flu[d$Year=='2008'] <- 0.79
d$flu[d$Year=='2009'] <- 1.06
d$flu[d$Year=='2010'] <- 1.3
d$flu[d$Year=='2011'] <- 4.0
d$flu[d$Year=='2012'] <- 1.1
d$flu[d$Year=='2013'] <- 2.4
d$flu[d$Year=='2014'] <- 3.0
d$flu[d$Year=='2015'] <- 1.9
d$flu[d$Year=='2016'] <- 1.85

d <- d %>% 
	group_by(Id) %>% 
	mutate(sdTime=sd(Time)) %>%
	mutate(meanTime=mean(Time))
	
d$ran_more_than_once <- d$num != 1

