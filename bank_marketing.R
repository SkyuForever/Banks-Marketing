library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)  
library("ipred")
library("rpart")
library("rpart.plot")
library(reshape2)
library(woeBinning)
library(Information)
library(scorecard)
library(corrplot)
library(corrgram)
library(UBL)
library(smotefamily)
library("MLmetrics")
library("class") 
#读取数据
setwd("E:/大学生活/学习/大三上/数据挖掘/期末报告/bank-additional")
Data <- read.csv("bank-additional.csv",sep=';',as.is = FALSE)
colnames(Data)[16:20] <- c("emp_rate", "cpi","cci","r3m","employed")
head(Data,2)
# 去除重复数据
Data=unique(Data)
# 描述性统计分析
str(Data)
summary(Data)
Data$y=as.factor(Data$y)
# 绘制产品购买情况
plot=ggplot(Data, aes(x = y)) +
  geom_bar(fill = "skyblue") +
  labs(title = "产品购买情况")
plot + theme(plot.title = element_text(hjust = 0.5))

# 检查缺失值
colSums(is.na(Data))

# 分类型数据可视化分析
object_columns <- names(Data)[sapply(Data, is.factor)][-11]
f <- melt(Data, id.vars='y', measure.vars=object_columns)
# 使用ggplot2创建图形
g <- ggplot(f, aes(x=value, fill=factor(y))) +
  facet_wrap(~variable, scales="free", ncol=3) +
  labs(x="",y="Count") +
  guides(fill=guide_legend(title="y")) +
  theme_minimal()
# 调用自定义的barplot函数
g + geom_bar(stat="count",position = "dodge") + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

# 数值型数据可视化分析
num_columns <- c('age', 'duration','campaign', 'pdays', 'previous', 'emp_rate', 
                 'cpi', 'cci', 'r3m', 'employed')
# 使用reshape2包进行数据的melting
f <- melt(Data, id.vars='y', measure.vars=num_columns)
# 使用ggplot2创建图形
ggplot(f, aes(x=value, fill=factor(y))) +
  geom_density(alpha=0.7) +
  facet_wrap(~variable, scales="free", ncol=3) +
  theme_minimal() +  # 修改为自己需要的主题
  labs(x="Value", y="Density") +  # 根据需要修改轴标签
  guides(fill=guide_legend(title="y"))  # 根据需要修改图例标签

# 可视化分析
g=ggplot(Data, aes(job))
g + geom_bar(aes(fill = y))+ylab('Count')+
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

ggplot(Data, aes(y = job)) +
  geom_bar(aes(fill = y), position ='fill') 

g <- ggplot(Data, aes(job))
position_dodge2(
  width = NULL,
  preserve = c("total", "single"),
  padding = 0.1,
  reverse = FALSE
)
g + geom_bar(aes(fill = y), position = position_dodge2())+ylab('intensity')+
  ggtitle('job')+
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        plot.title = element_text(hjust = 0.5))+ylab('Count')

g + geom_bar(aes(fill = y),
             position = 'fill')+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+ylab('intensity')

# 特征工程
## 构造特征all_previous
Data$all_previous <- Data$campaign + Data$previous
ggplot(Data, aes(x = all_previous)) + geom_boxplot()

Data=var_filter(Data, y = "y",positive = 'yes')
num_columns <- c('age', 'duration','campaign', 'previous', 'emp_rate', 
                 'cpi', 'cci', 'r3m', 'employed','all_previous')
Data=as.data.frame(Data)

# 箱形图
f <- melt(Data, id.vars='y', measure.vars=num_columns)
# 使用ggplot2创建箱线图
g <- ggplot(f, aes(x=variable, y=value,fill="#5ab4ac")) +
  geom_boxplot() +
  facet_wrap(~variable, scales="free", ncol=3,strip.position = "top") +
  labs(x="Variable", y="Value") +
  theme(strip.text = element_blank())# 根据需要修改主题
g
c=Data$cci<(-30)
Data=Data[c,]
#特征分箱
categorical_features=c('age', 'duration', 'campaign','previous','all_previous','y')
bins = woebin(Data[categorical_features], y="y",positive=1)
woebin_plot(bins)
bins

# age分箱
Data$age_bin <- cut(Data$age, breaks=c(-Inf, 34, 36, 48, 58, Inf), labels=c(0, 1, 2, 3, 4), right=FALSE)
Data$duration_bin <- cut(Data$duration, breaks=c(-Inf, 120, 160, 360, 680, Inf), labels=c(0, 1, 2, 3, 4), right=FALSE)
Data$campaign_bin <- cut(Data$campaign, breaks=c(-Inf, 4,6, Inf), labels=c(0, 1, 2), right=FALSE)
Data$previous_bin <- cut(Data$previous, breaks=c(-Inf, 1, Inf), labels=c(0, 1), right=FALSE)
Data$all_previous_bin <- cut(Data$all_previous, breaks=c(-Inf, 2,3,7, Inf), labels=c(0,1,2,3), right=FALSE)

all_iv <- sapply(bins, function(x){return(x$total_iv[1])})
Data[,all_iv < 0.015] <- NULL

# converting original value to woe
dt_woe = woebin_ply(Data[categorical_features], bins=bins)
str(dt_woe)
col<- c('job', 'marital', 'education','default','contact','month','poutcome',
        'emp_rate','cpi','cci','r3m','employed','campaign_bin', 
        'age_bin', 'duration_bin','previous_bin','all_previous_bin','y')
Data=Data[col]

# 处理分类型数据
## 有序变量
Data$education <- factor(Data$education, levels = c("unknown","illiterate", 
                                                    "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course", "university.degree"), ordered = TRUE)
# 将有序因子转换为整数编码
Data$education <- as.integer(Data$education)
Data$education <- Data$education-1
Data$contact <- ifelse(Data$contact == "cellular", 0, 1)
Data$contact <- as.integer(Data$contact)

num_columns=c("education","contact","emp_rate","cpi","cci","r3m","employed",
              "campaign_bin","age_bin","duration_bin","previous_bin","all_previous_bin",'y')
c=c("campaign_bin","age_bin","duration_bin","previous_bin","all_previous_bin")
for (object in c){
  Data[,object] = as.numeric(Data[,object])
  Data[,object] = Data[,object]-1
}
str(Data)
v<-cor(Data[,num_columns])
# 删除相关性高的特征
corrplot(v)
corrgram(Data[num_columns],lower.panel = panel.shade,upper.panel = panel.pie,text.panel = panel.txt)

col = c('job', 'marital', 'education','default','contact','month','poutcome',
       'cpi','cci','r3m','campaign_bin','age_bin', 'duration_bin',
       'previous_bin','all_previous_bin','y')

Data=Data[,col]

## 无序分类one-hot编码
object_col <- c(
  'job', 'marital','default','month','poutcome', 'campaign_bin','age_bin', 'duration_bin',
  'all_previous_bin')
c=c('campaign_bin','age_bin', 'duration_bin','all_previous_bin')
for (object in c){
  Data[,object] = as.factor(Data[,object])
}
Data[,'previous_bin']=Data[,'previous_bin']-1
Data_x <- model.matrix(~., data = Data[, object_col])
Data_x=Data_x[,-1]
colnames(Data_x)
Data=cbind(Data, Data_x)
Data=Data[, !(colnames(Data) %in% object_col)]

#SMOTE过采样
Data1=SMOTE(Data[,-7],Data[7],dup_size =0,K=5)
Data=Data1$data
colnames(Data)[47] <- 'y'
#标准化
col=c('cpi','cci','r3m')
Data[,col]=scale(Data[,col])

# 定义序列ind，随机抽取1和2,1的个数占80%，2的个数占20%
set.seed(12345)
ind <- sample(1:2, nrow(Data), replace = TRUE, prob = c(0.8, 0.2))
colnames(Data)[c(7,12)]=c("jobblue_collar","jobself_employed")
Data[,'y']=as.factor(Data[,'y'])
trainData <- Data[ind == 1, ]  # 训练数据
testData <- Data[ind == 2, ]  # 测试数据
# 随机打乱顺序
trainData=trainData[sample(nrow(trainData)), ]
testData=testData[sample(nrow(testData)), ]
#write.csv(trainData, "./trainData.csv", row.names = FALSE)
#write.csv(testData, "./testData.csv", row.names = FALSE)

# 随机森林
(rFM<-randomForest(y~.,data=trainData,importance=TRUE,proximity=TRUE))

head(rFM$votes)#各观测的各类别预测概率    
head(rFM$oob.times)#各观测作为OOB的次数
DrawL<-par()
par(mfrow=c(2,1),mar=c(5,5,3,1))
plot(rFM,main="随机森林的OOB错判率和决策树棵树")
plot(margin(rFM),type="h",main="边界点探测",xlab="观测序列",ylab="比率差") 
par(DrawL)
#选择150颗决策树
(rFM<-randomForest(y~.,data=trainData,ntree=150,importance=TRUE,proximity=TRUE))
Fit<-predict(rFM,testData)
(ConfM5<-table(testData$y,Fit))
F1_Score(Fit,testData$y)
(E5<-(sum(ConfM5)-sum(diag(ConfM5)))/sum(ConfM5))
1-E5

head(treesize(rFM))   
head(getTree(rfobj=rFM,k=1,labelVar=TRUE))
barplot(rFM$importance[,3],main="输入变量重要性测度(预测精度变化)指标柱形图")
box()
importance(rFM,type=1)
varImpPlot(x=rFM, sort=TRUE, n.var=nrow(rFM$importance),main="输入变量重要性测度散点图") 
col=c("duration_bin4","r3m","duration_bin3","cci","poutcomesuccess","cpi",
      "duration_bin2","education","maritalmarried","age_bin2","y")
trainData1=trainData[,col]
testData1=testData[,col]
#选择150颗决策树
(rFM<-randomForest(y~.,data=trainData1,ntree=150,importance=TRUE,proximity=TRUE))
Fit<-predict(rFM,testData1)
(ConfM5<-table(testData1$y,Fit))
F1_Score(Fit,testData1$y)
(E5<-(sum(ConfM5)-sum(diag(ConfM5)))/sum(ConfM5))
1-E5


# 支持向量机
set.seed(12345)
trainData$y=as.factor(trainData$y)
tObj<-tune.svm(y~.,data=trainData,type="C-classification",kernel="radial",
               cost=c(0.1,1,10),gamma=10^(-3:-1),scale=FALSE)
plot(tObj,xlab=expression(gamma),ylab="损失惩罚参数C",
     main="不同参数组合下的预测错误率",nlevels=10,color.palette=terrain.colors)
#tObj<-tune.svm(y~.,data=trainData,type="C-classification",kernel="radial",
#              cost=c(0.001,0.01,0.1,1),scale=FALSE)

summary(tObj)
BestSvm<-tObj$best.model
summary(BestSvm)
yPred<-predict(BestSvm,testData)
(ConfM<-table(yPred,testData$y))
(Err<-(sum(ConfM)-sum(diag(ConfM)))/sum(ConfM))
(F1_Score(yPred,testData$y))

# KNN分类
errRatio<-vector()   
for(i in 1:30){
  KnnFit<-knn(train=trainData[,-47],test=testData[,-47],cl=trainData[,47],k=i,prob=FALSE) 
  CT<-table(testData[,47],KnnFit) #计算混淆矩阵
  errRatio<-c(errRatio,(1-sum(diag(CT))/sum(CT))*100)    
}
plot(errRatio,type="b",xlab="近邻个数K",ylab="错判率(%)",main="预测中的近邻数K与错判率")

KnnFit<-knn(train=trainData[,-47],test=testData[,-47],cl=trainData[,47],k=1,prob=FALSE) 
(CT<-table(testData[,47],KnnFit)) #计算混淆矩阵
(Err<-(sum(CT)-sum(diag(CT)))/sum(CT))
1-Err
(F1_Score(KnnFit,testData$y))

#决策树
set.seed(12345)
Ctl=rpart.control(minsplit = 20,maxcompete = 4,maxdepth = 30,cp=0,xval = 10)
(TreeFit<-rpart(y~.,data=trainData,method="class",parms=list(split="gini"))) 
rpart.plot(TreeFit,type=4,branch=0,extra=2)#可视化决策树
printcp(TreeFit)#显示复杂度参数
plotcp(TreeFit)#可视化复杂度参数
(TreeFit<-rpart(y~.,data=trainData,method="class",cp=0,parms=list(split="gini"))) 
Fit<-predict(TreeFit,testData,type='class')
(ConfM3<-table(testData$y,Fit))
(E3<-(sum(ConfM3)-sum(diag(ConfM3)))/sum(ConfM3))
F1_Score(Fit,testData$y)

#bagging建立分类树
(BagM1<-bagging(y~.,data=trainData,nbagg=50,coob=TRUE))
CFit2<-predict(BagM1,testData,type="class")
ConfM2<-table(testData$y,CFit2)
(F1_Score(CFit2,testData$y))
(E2<-(sum(ConfM2)-sum(diag(ConfM2)))/sum(ConfM2))
F1_Score(CFit2,testData$y)
