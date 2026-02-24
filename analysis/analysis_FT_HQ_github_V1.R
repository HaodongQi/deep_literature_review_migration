
# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }

pacman::p_load(data.table, ggplot2, tidyverse)

####################################################################
# -- recode function

recodeFun <- function(df){
  df <- df |> as.data.frame()
  # for Q2
  df <- df %>% mutate(Q2=ifelse(Q2=="", "999", Q2))
  # FOR Q3
  df <- df %>% mutate(Q3=ifelse(Q3=="", "999", Q3))
  # FOR Q4
  df <- df %>% mutate(Q4=ifelse(Q4=="", "999", Q4))
  # %>%
  #   mutate(Q4=lapply(Q4, function(x) if(x=="0") x="42" else x=x) ) # None=0; Other=42
  # FOR Q5
  df <- df %>% mutate(Q5=ifelse(Q5=="", "999", Q5) )
}

# recodeFun <- function(df){
#   df <- df |> as.data.frame()
#   # for Q2
#   df <- df %>% mutate(Q2=paste(Q2, "999", sep=","))
#   # FOR Q3
#   df <- df %>% mutate(Q3=paste(Q3, "999", sep=","))
#   # FOR Q4
#   df <- df %>% mutate(Q4=paste(Q4, "999", sep=",")) 
#   # %>% 
#   #   mutate(Q4=lapply(Q4, function(x) if(x=="0") x="42" else x=x) ) # None=0; Other=42
#   # FOR Q5
#   df <- df %>% mutate(Q5=paste(Q5, "999", sep=",") ) 
# }

####################################################################
#-- Data
# Coded
coded <- fread("allcoded.csv", sep="\t",  colClasses = "character")
## check duplicates
table(duplicated(coded$id))
coded <- coded[-which(duplicated(coded$id)),]
table(duplicated(coded$id))
coded <-  recodeFun(coded)

# Initial LLM predicted label
llmLab <- fread("classification_verification-L32-3B.csv", colClasses = "character", fill=TRUE)
llmLab <- llmLab[complete.cases(llmLab)]
for(i in 1:6){
  var <- sprintf("Q%d", i)
  llmLab[, (var) := gsub("\\[", "", get(var))]
  llmLab[, (var) := gsub("\\]", "", get(var))]
}
table(duplicated(llmLab$id))
if(any(duplicated(llmLab$id))){
  llmLab <- llmLab[-which(duplicated(llmLab$id)),]
}
table(duplicated(llmLab$id))
idx <- which(llmLab$Q1=="")
if(length(idx)>0){
  llmLab <- llmLab[-idx,]
}
llmLab <-  recodeFun(llmLab)

# Note: Initial AI crowd label (llmLab) contains no missing data.
# However, coded label may contain missing data if ALL AI crouwd labels are rejected.
# Hence, training and testing set may also contain missing data.
# These missing data should be treated as correct label, 
# as the model has learned not to predict those rejected by coders.
# i.e., not predicting = all rejected by coders
llmLab$Q5 %>% table()
coded$Q5 %>% table()

####################################################################
#--  binary 

# metric function
binary_acc_fun <- function(predVec, trueVec) {
  matches <- predVec == trueVec 
  matches <- as.numeric(matches) 
  matches <- mean(matches)
  return(matches)
}

#  classification function
binaryClassFun <- function(coded_df, pred_df, whichQ){
  temp <- coded_df |> as.data.frame()
  temp <- data.frame(id=temp$id, seq=temp[,whichQ])
  temp$seq <- as.numeric(temp$seq)
  trueSeq <- temp |> mutate(trueVal=seq) |> 
    dplyr::select(id,trueVal)
  
  temp <- pred_df |> as.data.frame()
  temp <- data.frame(id=temp$id, seq=temp[,whichQ])
  temp$seq <- as.numeric(temp$seq)
  predSeq <- temp |> mutate(predVal=seq) |> 
    dplyr::select(id,predVal)
  
  metric <- left_join(predSeq,trueSeq, by="id") |> 
    mutate(ACC=mapply(binary_acc_fun, predVal, trueVal) |> as.numeric() ) 
  metric <- data.frame(value=mean(metric$ACC,na.rm = T), 
                       Metric="ACC", N = nrow(metric), Question=whichQ) 
  
  return(metric)
}

####################################################################
#-- multi-label
# jaccard metric function
multi_jac_fun <- function(predSeq, trueSeq) {
  predSeq <- unlist(predSeq)
  trueSeq <- unlist(trueSeq)
  intersection<- intersect(predSeq, trueSeq)
  leng_intersect <- length(intersection)
  union <- union(predSeq, trueSeq)
  leng_union <- length(union)
  # FN and TN
  similarity <- (leng_intersect/leng_union)
  return(similarity)
}

# classification function
multiClassFun <- function(coded_df, pred_df, whichQ){
  trueSeq <- coded_df |> as.data.frame() 
  temp <- sapply(trueSeq[, whichQ], function(x) strsplit(x, ",")) 
  temp <- lapply(temp, as.numeric) 
  trueSeq <- trueSeq |> mutate(trueVal=temp) |> 
    dplyr::select(id,trueVal)
  
  predSeq <- pred_df |> as.data.frame()
  temp <- sapply(predSeq[, whichQ], function(x) strsplit(x, ",")) 
  temp <- lapply(temp, as.numeric)
  predSeq <- predSeq |> mutate(predVal=temp) |> 
    dplyr::select(id,predVal)
  
  metric <- c()
  metric_df <- left_join(predSeq,trueSeq, by="id") |> 
    filter(trueVal!="NULL" & predVal!="NULL") 
  temp <- metric_df %>% 
    mutate(JAC=mapply(multi_jac_fun, 
                      predSeq=predVal, trueSeq=trueVal) |> as.numeric() )
  temp <- data.frame(value=mean(temp$JAC, na.rm = T), 
                     Metric="JAC", N = nrow(temp), Question=whichQ )
  metric <- rbind(metric, temp)
  return(metric)
}

####################################################################
#-- initial prediction
output <- c()
# Q1
temp <- binaryClassFun(coded, llmLab, "Q1")
temp$set <- "Train"
output <- rbind(output, temp)
# Q2
temp <- multiClassFun(coded_df=coded, pred_df=llmLab, whichQ="Q2")
temp$set <- "Train"
output <- rbind(output, temp)
# Q3
temp <- multiClassFun(coded_df=coded, pred_df=llmLab, whichQ="Q3")
temp$set <- "Train"
output <- rbind(output, temp)
# Q4
temp <- multiClassFun(coded_df=coded, pred_df=llmLab, whichQ="Q4")
temp$set <- "Train"
output <- rbind(output, temp)
# Q5
temp <- multiClassFun(coded_df=coded, pred_df=llmLab, whichQ="Q5")
temp$set <- "Train"
output <- rbind(output, temp)
# Q6
temp <- binaryClassFun(coded,llmLab,"Q6")
temp$set <- "Train"
output <- rbind(output, temp)
# All results
output <- output |> mutate(Split=1.02)
base <- output

####################################################################
#-- sample for 50% - 50% splitting
## Data
trainFT <- fread("classifications-train-FT.csv", colClasses = "character", fill=TRUE)
trainFT <- trainFT[complete.cases(trainFT)]
for(i in 1:6){
  var <- sprintf("Q%d", i)
  trainFT[, (var) := gsub("\\[", "", get(var))]
  trainFT[, (var) := gsub("\\]", "", get(var))]
}
trainFT[, 1:6] <- lapply(trainFT[, 1:6], function(x) gsub(" ", ",", x) )

testFT <- fread("classifications-test-FT.csv", colClasses = "character", sep =",", fill=TRUE)
testFT <- testFT[complete.cases(testFT)]
for(i in 1:6){
  var <- sprintf("Q%d", i)
  testFT[, (var) := gsub("\\[", "", get(var))]
  testFT[, (var) := gsub("\\]", "", get(var))]
}
testFT[, 1:6] <- lapply(testFT[, 1:6], function(x) gsub(" ", ",", x) )

# check duplicates
table(duplicated(trainFT$id))
if(any(duplicated(trainFT$id))){
  trainFT <- trainFT[-which(duplicated(trainFT$id)),]
}
table(duplicated(trainFT$id))
idx <- which(trainFT$Q1=="")
if(length(idx)>0){
  trainFT <- trainFT[-idx,]
}

table(duplicated(testFT$id))
if(any(duplicated(testFT$id))){
  testFT <- testFT[-which(duplicated(testFT$id)),]
}
table(duplicated(testFT$id))
idx <- which(testFT$Q1=="")
if(length(idx)>0){
  testFT <- testFT[-idx,]
}

# recode
trainFT <- recodeFun(trainFT)
testFT <- recodeFun(testFT)

output <- c()
# Q1
temp <- binaryClassFun(coded, trainFT, "Q1")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- binaryClassFun(coded, testFT, "Q1")
temp$set <- "Test"
output <- rbind(output, temp)
# Q2
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q2")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q2")
temp$set <- "Test"
output <- rbind(output, temp)
# Q3
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q3")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q3")
temp$set <- "Test"
output <- rbind(output, temp)
# Q4
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q4")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q4")
temp$set <- "Test"
output <- rbind(output, temp)
# Q5 
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q5")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q5")
temp$set <- "Test"
output <- rbind(output, temp)
# Q6 
temp <- binaryClassFun(coded, trainFT, "Q6")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- binaryClassFun(coded, testFT, "Q6")
temp$set <- "Test"
output <- rbind(output, temp)
# All results
output <- output |> mutate(Split=.5)
spl50 <- output

####################################################################
# same for 80% - 20% splitting
## Data
trainFT <- fread("classifications-80-train-FT.csv", colClasses = "character", fill=TRUE)
trainFT <- trainFT[complete.cases(trainFT)]
for(i in 1:6){
  var <- sprintf("Q%d", i)
  trainFT[, (var) := gsub("\\[", "", get(var))]
  trainFT[, (var) := gsub("\\]", "", get(var))]
}
trainFT[, 1:6] <- lapply(trainFT[, 1:6], function(x) gsub(" ", ",", x) )

testFT <- fread("classifications-80-test-FT.csv", colClasses = "character", sep =",", fill=TRUE)
testFT <- testFT[complete.cases(testFT)]
for(i in 1:6){
  var <- sprintf("Q%d", i)
  testFT[, (var) := gsub("\\[", "", get(var))]
  testFT[, (var) := gsub("\\]", "", get(var))]
}
testFT[, 1:6] <- lapply(testFT[, 1:6], function(x) gsub(" ", ",", x) )

# check duplicates
table(duplicated(trainFT$id))
if(any(duplicated(trainFT$id))){
  trainFT <- trainFT[-which(duplicated(trainFT$id)),]
}
table(duplicated(trainFT$id))
idx <- which(trainFT$Q1=="")
if(length(idx)>0){
  trainFT <- trainFT[-idx,]
}

table(duplicated(testFT$id))
if(any(duplicated(testFT$id))){
  testFT <- testFT[-which(duplicated(testFT$id)),]
}
table(duplicated(testFT$id))
idx <- which(testFT$Q1=="")
if(length(idx)>0){
  testFT <- testFT[-idx,]
}

# recode
trainFT <- recodeFun(trainFT)
testFT <- recodeFun(testFT)

output <- c()
# Q1
temp <- binaryClassFun(coded, trainFT, "Q1")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- binaryClassFun(coded, testFT, "Q1")
temp$set <- "Test"
output <- rbind(output, temp)
# Q2
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q2")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q2")
temp$set <- "Test"
output <- rbind(output, temp)
# Q3
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q3")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q3")
temp$set <- "Test"
output <- rbind(output, temp)
# Q4
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q4")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q4")
temp$set <- "Test"
output <- rbind(output, temp)
# Q5 
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q5")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q5")
temp$set <- "Test"
output <- rbind(output, temp)
# Q6 
temp <- binaryClassFun(coded, trainFT, "Q6")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- binaryClassFun(coded, testFT, "Q6")
temp$set <- "Test"
output <- rbind(output, temp)
# All results
output <- output |> mutate(Split=.8)
spl80 <- output

####################################################################
# same for 95% - 5% splitting
## Data
trainFT <- fread("classifications-95-train-FT.csv", colClasses = "character", fill=TRUE)
trainFT <- trainFT[complete.cases(trainFT)]
for(i in 1:6){
  var <- sprintf("Q%d", i)
  trainFT[, (var) := gsub("\\[", "", get(var))]
  trainFT[, (var) := gsub("\\]", "", get(var))]
}
trainFT[, 1:6] <- lapply(trainFT[, 1:6], function(x) gsub(" ", ",", x) )

testFT <- fread("classifications-95-test-FT.csv", colClasses = "character", sep =",", fill=TRUE)
testFT <- testFT[complete.cases(testFT)]
for(i in 1:6){
  var <- sprintf("Q%d", i)
  testFT[, (var) := gsub("\\[", "", get(var))]
  testFT[, (var) := gsub("\\]", "", get(var))]
}
testFT[, 1:6] <- lapply(testFT[, 1:6], function(x) gsub(" ", ",", x) )

# check duplicates
table(duplicated(trainFT$id))
if(any(duplicated(trainFT$id))){
  trainFT <- trainFT[-which(duplicated(trainFT$id)),]
}
table(duplicated(trainFT$id))
idx <- which(trainFT$Q1=="")
if(length(idx)>0){
  trainFT <- trainFT[-idx,]
}

table(duplicated(testFT$id))
if(any(duplicated(testFT$id))){
  testFT <- testFT[-which(duplicated(testFT$id)),]
}
table(duplicated(testFT$id))
idx <- which(testFT$Q1=="")
if(length(idx)>0){
  testFT <- testFT[-idx,]
}

# recode
trainFT <- recodeFun(trainFT)
testFT <- recodeFun(testFT)

output <- c()
# Q1
temp <- binaryClassFun(coded, trainFT, "Q1")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- binaryClassFun(coded, testFT, "Q1")
temp$set <- "Test"
output <- rbind(output, temp)
# Q2
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q2")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q2")
temp$set <- "Test"
output <- rbind(output, temp)
# Q3
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q3")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q3")
temp$set <- "Test"
output <- rbind(output, temp)
# Q4
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q4")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q4")
temp$set <- "Test"
output <- rbind(output, temp)
# Q5 
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q5")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- multiClassFun(coded_df=coded, pred_df=testFT, whichQ="Q5")
temp$set <- "Test"
output <- rbind(output, temp)
# Q6 
temp <- binaryClassFun(coded, trainFT, "Q6")
temp$set <- "Train"
output <- rbind(output, temp)
temp <- binaryClassFun(coded, testFT, "Q6")
temp$set <- "Test"
output <- rbind(output, temp)
# All results
output <- output |> mutate(Split=.95)
spl95 <- output

####################################################################
# same for 100% 
## Data
trainFT <- fread("classifications-100-train-FT.csv", colClasses = "character", fill=TRUE)
trainFT <- trainFT[complete.cases(trainFT)]
for(i in 1:6){
  var <- sprintf("Q%d", i)
  trainFT[, (var) := gsub("\\[", "", get(var))]
  trainFT[, (var) := gsub("\\]", "", get(var))]
}
trainFT[, 1:6] <- lapply(trainFT[, 1:6], function(x) gsub(" ", ",", x) )

# check duplicates
table(duplicated(trainFT$id))
if(any(duplicated(trainFT$id))){
  trainFT <- trainFT[-which(duplicated(trainFT$id)),]
}
table(duplicated(trainFT$id))
idx <- which(trainFT$Q1=="")
if(length(idx)>0){
  trainFT <- trainFT[-idx,]
}

# recode
trainFT <- recodeFun(trainFT)
testFT <- recodeFun(testFT)

output <- c()
# Q1
temp <- binaryClassFun(coded, trainFT, "Q1")
temp$set <- "Train"
output <- rbind(output, temp)
# Q2
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q2")
temp$set <- "Train"
output <- rbind(output, temp)
# Q3
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q3")
temp$set <- "Train"
output <- rbind(output, temp)
# Q4
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q4")
temp$set <- "Train"
output <- rbind(output, temp)
# Q5 
temp <- multiClassFun(coded_df=coded, pred_df=trainFT, whichQ="Q5")
temp$set <- "Train"
output <- rbind(output, temp)
# Q6 
temp <- binaryClassFun(coded, trainFT, "Q6")
temp$set <- "Train"
output <- rbind(output, temp)
# All results
output <- output |> mutate(Split=1)
spl100 <- output

####################################################################

# plotdf <- rbind(spl50,spl80,spl100) |> 
#   gather(Metric,value,c(OneMinusHL,JAC,ACC)) |> drop_na()
# ggplot(plotdf, aes(x=Split*100,y=value*100), alpha=.6) +
#   facet_wrap(Question~., ncol=2) +
#   geom_line(aes(color=Metric, linetype=set)) +
#   geom_point(aes(color=Metric), shape=1) +
#   labs(x="Training size in %", y="Accuracy in %",
#       linetype = "Set")
# ggsave("temp.png", width = 6, height = 6)

plotdf <- rbind(spl50,spl80, spl95, base) # spl100,
write.csv(plotdf, "ftAccuracy.csv", row.names = F)

plotdf <- read.csv("ftAccuracy.csv") %>% 
  group_by(Question,Split,Metric) %>% 
  dplyr::mutate(TrainN=max(N)) %>% 
  mutate(qLabel=case_when(
    Question=="Q1" ~ "1. About migration & mobility?",
    Question=="Q2" ~ "3. What migration drivers discussed?",
    Question=="Q3" ~ "Sentiment",
    Question=="Q4" ~ "4. What clim. & env. hazards discussed?",
    Question=="Q5" ~ "Discipline",
    Question=="Q6" ~ "2. Used quant. methods?",
    TRUE ~ Question))

plotdf <- plotdf %>% filter(!grepl("Senti",qLabel) & !grepl("Disc",qLabel)) 

ggplot(plotdf |> filter(Split<=1), aes(x=TrainN,y=value*100), alpha=.6) +
  facet_wrap(qLabel~., ncol=2) +
  geom_line(aes(color=set), linetype="dashed") +
  geom_point(aes(color=set), size=2.8, shape=1) +
  labs(x="Training size N", y="Accuracy in %",
       linetype = "Set") +
  geom_hline(
    data=plotdf |> filter(Split>1),
    aes(yintercept = value*100), color="grey50", linetype="dashed"
  ) +
  ggrepel::geom_text_repel(
    data=plotdf |> filter(Split>1),
    aes(x=TrainN,y=value*100), color="grey50", 
    label="Initial Label", size=3.8, show.legend=F, direction = "y") +
  geom_text(
    aes(x=min(plotdf$TrainN)*1.1, y=max(plotdf$value*100)*1,
    label=paste0("Metric: ",Metric) ), 
    size=3.8) +
  theme_bw()

ggsave("metric.png", width = 8, height = 6)

export_tab <- plotdf |> filter(Split<=1) |> ungroup() |> select(value, Metric, set, TrainN, qLabel) |> 
  mutate(value=paste0(round(value*100), "%"))
export_tab <- export_tab |> pivot_wider(
  id_cols    = c(qLabel, set, Metric),
  names_from = TrainN,
  names_prefix = "N=",
  values_from = value,
  names_sort = TRUE
)

temp <- plotdf |> filter(Split>1) |> ungroup() |> select(value, qLabel, set, Metric) |> 
  mutate(value=paste0(round(value*100), "%")) |> 
  dplyr::rename(Init.Label=value)
export_tab <- export_tab |> left_join(temp) |> arrange(qLabel) |> 
  mutate(across(everything(), ~ ifelse(is.na(.x), "--", .x)))

pacman::p_load(knitr)
kable(
  export_tab,
  format = "latex",
  booktabs = TRUE,
  escape = TRUE
)


