#-----------Direct_Marketing_Linear Regression------------#
# To predict the amount spent by the customer using a Linear Regression Model #

setwd("C:\\Users\\Varun R Bhat\\Downloads\\Jigsaw\\Direct_Marketing_Data_Set")

library(dplyr)
library(ggplot2)
library(car)

#Reading the dataset

data<-read.csv("directmarketing.csv")

#Exploratory Data Analysis

head(data)
str(data)

#Age
plot(data$Age,data$AmountSpent,col="red") #Boxplot

#Middle and Old levels behave in a similar way and can be combined
data$Age1<-ifelse(data$Age!="Young","Middle-Old",as.character(data$Age))
data$Age1<-as.factor(data$Age1)
summary(data$Age1)

plot(data$Age1,data$AmountSpent,col="red")

#Gender
plot(data$Gender,data$AmountSpent,col="red")

#OwnHome
plot(data$OwnHome,data$AmountSpent,col="red")

#Married
plot(data$Married,data$AmountSpent,col="red")

#Location
plot(data$Location,data$AmountSpent,col="red")

#Salary
plot(data$Salary,data$AmountSpent) #Scatter plot

#Children
summary(data$Children)
unique(data$Children) #can be converted into factor since there are 4 levels

data$Children<-as.factor(data$Children)

plot(data$Children,data$AmountSpent,col="red")
#Levels 2 and 3 are similar and therefore can be combined

data$Children1<-ifelse(data$Children=="2"|data$Children=="3","2-3",as.character(data$Children))
data$Children1<-as.factor(data$Children1)
summary(data$Children1)

plot(data$Children1,data$AmountSpent,col="red")

#History
summary(data$History)
#Summarize amount spent based on History 
tapply(data$AmountSpent,data$History,mean)

#Check mean amount spent for the missing value group
ind<-which(is.na(data$History))
mean(data[ind,"AmountSpent"])

#Treat the missing values as a separate group since it is not similar to any other group

data$History1<-ifelse(is.na(data$History),"Missing",as.character(data$History))
data$History1<-as.factor(data$History1)

summary(data$History1)
plot(data$History1,data$AmountSpent,col="red")

#Catalogs
unique(data$Catalogs)
summary(data$Catalogs)
data$Catalogs<-as.factor(data$Catalogs)

data1<-data[,-c(1,7,8,11)]

#Linear Regression Model

mod1<-lm(AmountSpent~.,data = data1)

summary(mod1)

mod2<-lm(formula = AmountSpent~ Gender + Location + Salary + Catalogs + Children1
         + History1, data = data1)

summary(mod2)

#Creating dummy variables

data1$Male_d<-ifelse(data1$Gender=="Male",1,0)
data1$Female_d<-ifelse(data1$Gender=="Female",1,0)

data1$Low_d<-ifelse(data1$History1=="Low",1,0)
data1$Med_d<-ifelse(data1$History1=="Medium",1,0)

mod3<-lm(formula = AmountSpent~ Male_d + Location + Salary + Catalogs + Children1
         + Low_d + Med_d, data = data1)
summary(mod3)

mod4<-lm(formula = AmountSpent~ Location + Salary + Catalogs + Children1
         + Low_d + Med_d, data = data1)
summary(mod4)

#Model assumptions check
# Residuals distribution should be normal

hist(mod4$residuals) #+ve skew
qqPlot(mod4$residuals) #non normal distribution

#Multicollinearity Check

vif(mod4) #Ok

#Check for Heteroscedasticity

plot(mod4$fitted.values,mod4$residuals) #funnel shape - heteroscedasticity is observed

#Remedies: Apply log transform to y variable

mod5<-lm(formula = log(AmountSpent)~ Location + Salary + Catalogs + Children1
         + Low_d + Med_d, data = data1)
summary(mod5)

qqPlot(mod5$residuals) #Slightly better
vif(mod5) #OK
plot(mod5$fitted.values,mod5$residuals) #OK

#Fit chart

predicted<-mod5$fitted.values
actual<-log(data1$AmountSpent)

dat<-data.frame(predicted,actual)

p<-ggplot(dat,aes(x=row(dat)[,2],y=predicted))
p+geom_line(colour="blue")+geom_line(data = dat,aes(y=actual),colour="black")

#---------------------------------------------------------------------------------#