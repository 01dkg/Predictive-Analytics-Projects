#This script will create a decriptive statistics table for oj model for some variables
#######################################################################################

#Creating a subset of required number of columns from the dataset 
df <- unique(oj[c(7:17)]) 

#Generating a list of subset of columns 
Des_stats <- do.call(data.frame, 
             list(Average = apply(df, 2, mean),
                  Standard_Deviation = apply(df, 2, sd),
                  Median = apply(df, 2, median),
                  Minimum = apply(df, 2, min),
                  Maximum = apply(df, 2, max)))

print(Des_stats)

#Correlation 
corr = apply(df,2,quantile,probs=c(0.05,.1,.5,.95,.99))
print(corr)
corr1 <- cor(df,use = "complete.obs",method="kendall")
print(round(corr1,digits=3))

pairs(df,panel = points,main="Correlation plot among variables of dataset")
######################################################################################
store <- tapply(oj$logmove, INDEX = list(oj$store,brands), FUN = sum, na.rm =TRUE)
df2 <- cbind(store)
write.table(df2, file = "store.csv",sep = ",",row.names = T)
library(readr)
sdata <- read_csv("G:/R_programs_git/R_Progams/store.csv", col_names = FALSE, skip = 1)
ggplot(sdata, aes(sdata$X1)) + 
  geom_point(aes(y = sdata$X2, color = "Dominicks")) +
  geom_point(aes(y = sdata$X3, color = "Minute Maid")) + 
  geom_point(aes(y = sdata$X4, color = "Tropicana")) + xlab("Store") + ylab("Number of units sold")
