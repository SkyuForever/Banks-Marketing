# Banks-Marketing
目前传统的银行营销方式已逐渐显露出颓势，实现精准营销成为银行业面临的重要问题。选取UCI银行营销数据集作为研究对象，并进行数据清洗、数据分析和特征工程。同时利用SMOTE过采样解决数据集不平衡问题。最后使用随机森林、KNN、SVM等多种机器学习算法，构建了一个精确且稳健的银行精准营销预测模型。

------

## 1.解题思路

我们通过构建合适的指令，让大模型对相应的题目进行作答。同时收集电子、物理、化学、数学等学科的背景知识以及注册电气工程师的参考资料和题库，利用langchain框架，先以检索的方式检索出和当前问题相关的知识，再作为背景知识输入到模型，帮助模型进行作答。

## 2.代码说明

### 2.1 代码文件结构
```
├── document_loaders
├── knowledge
├── testsplitter
├── test.py
├── setup.py
├── predict.py
├── question.json
├── requirements.txt
└── README.md
```
其中 document_loaders 和 testsplitter 来自项目[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)

### 2.2 代码文件说明

``` 
* test.py                       模型QA文件   
* setup.py                      数据格式转换文件   
* predict.py                    结果文件转换文件    
* knowledge                     使用的知识库
* requirements.txt              运行环境要求    
* question.json                 进行QA问答的数据集          
```

## 3.运行说明

### 3.1运行环境

* python == 3.10.12

### 3.3运行步骤
```
# 1. 运行setup.py文件，对question.json文件格式进行调整
$ python setup.py

# 2. 运行test.py文件，对调整后的问答数据集
$ python test.py

# 3. 运行predict.py文件，将生成的结果文件格式调整为json格式
$ python predict.py
```

## 4.结果与改进
