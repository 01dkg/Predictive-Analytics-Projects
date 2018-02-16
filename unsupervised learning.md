---
title: "R Notebook"
output: html_notebook
---

```{r}
# Load data
data("USArrests")
my_data <- USArrests
# Remove any missing value (i.e, NA values for not available)
my_data <- na.omit(my_data)
# Scale variables
my_data <- scale(my_data)
# View the firt 3 rows
head(my_data, n = 3)
```
```{r}
library("cluster")
library("factoextra")
res.dist <- get_dist(USArrests, stand = TRUE, method = "pearson")
fviz_dist(res.dist, 
   gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
```

```{r}
fviz_nbclust(my_data, kmeans, method = "gap_stat")
km.res <- kmeans(my_data, 4, nstart = 25)
fviz_cluster(km.res, data = my_data, frame.type = "convex")+
  theme_minimal()
```
# PAM clustering: Partitioning Around Medoids
```{r}
# Compute PAM
library("cluster")
pam.res <- pam(my_data, 4)
# Visualize
fviz_cluster(pam.res)
```
## Hierarchical clustering 
```{r}
# 1. Loading and preparing data
data("USArrests")
my_data <- scale(USArrests)
# 2. Compute dissimilarity matrix
d <- dist(my_data, method = "euclidean")
# Hierarchical clustering using Ward's method
res.hc <- hclust(d, method = "ward.D2" )
# Cut tree into 4 groups
grp <- cutree(res.hc, k = 4)
# Visualize
plot(res.hc, cex = 0.6) # plot tree
rect.hclust(res.hc, k = 4, border = 2:5) # add rectangle
```



