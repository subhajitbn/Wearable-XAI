require("sirus")

features_for_sirus <- read.csv("C:/Users/SAMSUNG/Desktop/Github/Wearable-XAI/output/features_for_sirus.csv")

data <- as.data.frame(features_for_sirus[, -ncol(features_for_sirus)])
y <- features_for_sirus$label
sirus.m <- sirus.fit(data, y)
sirus.m
sirus.print(sirus.m)
