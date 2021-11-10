#Good practice: clear your workspace before you get started. 
rm(list = ls())

library(rpart)
library(rpart.plot)
library(ggplot2)
library(pROC)
library(RColorBrewer)

#read in data (fraud_data.csv)
f <- read.csv("Class Datasets/fraud_data.csv", stringsAsFactors=TRUE)



### --------- EXPLORATORY ANALYSIS -------------

#Initial exploratory analysis 
#look at a summary of the data
summary(f)

#look at the structure
str(f)

#check out see if there are any character variables
#Note: character variables are easier to CLEAN but they can't be
#input to trees or forests as easily as factors 

#doesn't appear to be any issues with reading in the data 
#let's do some exploratory analysis to take a closer look
#(real life: do all variables, but in class, we'll just do a few)

#Months as customer (this is numeric) ---> make a histogram
ggplot(data = f) +
  geom_histogram(aes(x=months_as_customer))
#seems reasonable

#make a histogram of customer age 
ggplot(data = f) +
  geom_histogram(aes(x=age), binwidth = 1)
#density plot: an alternative to histograms for numeric data!
#(it approximates the PDF!)
ggplot(data = f) +
  geom_density(aes(x=age))

# incident type (this is a factor) - make a barchart
ggplot(data=f) +
  geom_bar(aes(x=incident_type))
#look how scrunched the x axis is - make it better
#sometimes when the x axis has really looooong variable names,
#it makes sense to "flip" the coordinate syetem
ggplot(data=f) +
  geom_bar(aes(x=incident_type)) +
  coord_flip() +
  labs(x = "Incident Type", y="Count of Claims")
#makes it so the variable names don't overlap on each other 
#since they're now vertical rather than horizontal

#now, we've explored some of our x variables in a univariate sense
#next, let's incorporate our y variable

#See how incident type impacts the probability of fraud being reported
#Use a filled barchat tp plot proportions!
ggplot(data=f) +
  geom_bar(aes(x=incident_type, fill = fraud_reported), position ="fill")+
  coord_flip() + 
  labs (x="Incident Type", y="Proportion of Claims") +
  scale_fill_grey("Fraud\nReported")

#Black / White / Grey is always color blind friendly
#Alternatively: RColorBrewer provides many colorblind friendly pallettes
#to choose from 

#See the palette choices:
#gives the names of the colorblind friendly palettes
display.brewer.all(colorblindFriendly = TRUE)

ggplot(data=f) +
  geom_bar(aes(x=incident_type, fill = fraud_reported), position ="fill")+
  coord_flip() + 
  labs (x="Incident Type", y="Proportion of Claims") +
  scale_fill_brewer("Fraud\nReported", palette = "RdPu")+
  theme_bw() #removes grey background
#above, within scale_fill_brewer() enter your chosen palette!



### --- DATA PREPARATION -------------

#Prepare the data and fir a tuned classification tree
#split data 80/20
#seed = 1981536 for splitting your data 
#seed = 172172172 for fitting tree - (need to set each time!)
RNGkind(sample.kind = "default")
set.seed(1981536)
train.idx <- sample(x = 1:nrow(f), size = floor(0.8*nrow(f)))
#make training data
train.df <- f[train.idx,]
#the rest will be for testing
test.df <- f[-train.idx,]

#fit the tree
set.seed(172172172) #for reproducibility
ftree<- rpart(fraud_reported ~ .,
              data = train.df,
              method = 'class')


#--------------------------------

#Print and plot the default tree
#note the importance of variables is reflected in the order in which they are used to split
#thus, the first few variables should be the most important in predicting fraud. 

#print classification rules
ftree
#plot tree
rpart.plot(ftree)
#this tree suggests that incident_severity and insured_education_level are
#some of the most important for prediction fraud

#create a visualization 
rpart.plot(ftree)
#the most important are incident_severity and insured_education_level

#visualize
ggplot(data = f) +
  geom_bar(aes(x = incident_severity, fill = fraud_reported), position = "fill") +
  facet_wrap(~insured_education_level) +
  labs(x = "Severity of Incident", y = "Proportion") + 
  coord_flip() +
  scale_fill_brewer("Fraud\nReported", palette = "BuGn")+
  theme_bw() #removes grey background

#The tree informed us as what variables might be most important 
#and worth incorporating into the plots. IN this example, major  is
#consistently associated with a higher probability
#of reported fraud. However, this is even more true for
#individuals listed as having a PhD or Jd. It is less true for
#individuals listed as having a highschool degree. 
#Further, a PhD with a total loss is less likely to be reported as fraudulent
#than other degrees with a total loss. 



###-------------------------------------
#re-fit tree, grown very large 
set.seed(172172172)
ftree <- rpart(fraud_reported~ .,
               data = train.df,
               method = "class",
               control = rpart.control(cp = 0.0001, minsplit = 1))
ftree
#This tree is huge(many terminal nodes)
#this would likely be a terrible predictor because it would overfit the data
#need to prune it back!
#find the sub tree within this huge tree that minimizes cross validation error:
printcp(ftree)
#2  0.0233333      1     0.790  0.790 0.056302
#because it has the smallest xerror

optimalcp <- ftree$cptable[which.min(ftree$cptable[,"xerror"]),"CP"]

ftree2 <- prune(ftree, cp = optimalcp)
#ftree2 is our final/pruned tree that we will use going forward


###--------------------------------------
rpart.plot(ftree2)
#the only variable involved is incident_severity
##this suggests that any further splits will only increase our
#cross validation error (an approx. of out-of-sample error)


###--------------------------------------
#Need to use an ROC curve to quantify performance of model
#in out of sample prediction
#First specify positive event: "Y" here
pi_hat <- predict(ftree2, test.df, type = "prob")[,"Y"] #choose Y: positive

rocCurve <- roc(response = test.df$fraud_reported, 
                predictor = pi_hat, 
                levels = c("N", "Y")) #first negative, then positive

#plot ROC curve:
plot(rocCurve, print.thres = TRUE, print.auc = TRUE)
#pi* = 0.353
#specificity = 0.895
#sensitivity = 0.681
#AUC = 0.788

#If we set our threshold at 0.353,
#When fraud is not being committed, we are going to
#predict that it is not being committed about 89.5% of the time. (specificity)
#When fraud IS being committed, we are going to 
#correctly catch/identify the fraud about 68.1% of the time. 

#Note, again, that this only holds if we set pi* = 0.353,
#that is, predict Y when our pi_hat > 0.353



###------------------------------------
pi_star_cutoff <- coords(rocCurve, x = "best")$threshold
#predict on new/test data using that cutoff above

test.df$fraud_prediction <- ifelse(pi_hat > pi_star_cutoff, "Y", "N")
#Let's use the strengths of GLM (interpretations, formal inference)
#with the strengths of trees (automatic variable selection)



###---------------------------------------------
#Our Bernoulli variable is 1 if fraud, 0 otherwise
f$fraud_binary <- ifelse(f$fraud_reported == "Y", 1, 0)

#fit a logistic regression only using the variables suggested
#by the classification tree (using xerror as the guide)

m <- glm(fraud_binary~incident_severity , data = f, 
         family = binomial(link = logit))
summary(m)#just like SAS's "Analysis of Maximum Likelihood Estiamtes" table
coef(m)
exp(coef(m))
#Interpret: total loss vs major damage
#The odds of a claim being fraudulent when the reported incident severity
#is total loss are exp(-2.3402952) = 0.09629920 TIMES the odds
#when incident severity is major damage. That is, they're about
#90% smaller for total loss claims. 

#The other way around: the odds of a claim being fraudulent
#when the reported incident severity is major damage
#are exp(-(-2.3402952)) = 10.3843 times the odds
#when incident severity is total loss. 

confint(m) #defaults to likelihood-based CI
exp(confint(m))#on the scale of odds ratios

#Interpret: minor damage vs total loss:

#the odds of a claim being fraudulent
#when the reported incident severity is minor damage
#are exp(beta_minor - beta_total) = 0.8150492 times the odds
#when incident severity is total loss. 
#or
# the odds of a claim being fraudulent
#when the reported incident severity is total loss
#are exp(beta_total - beta_minor) = 1.22692 times the odds
#when incident severity is total loss. 




