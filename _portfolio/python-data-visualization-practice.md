---
title: "Python数据可视化实战"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/python-data-visualization-practice
date: 2025-05-20
excerpt: "使用Matplotlib和Seaborn实现数据可视化，包括探索性分析图表和模型评估图表，提升数据分析结果的可读性。"
header:
  teaser: /images/portfolio/python-data-visualization-practice/age_distribution.png
tags:
- 数据可视化
- Python
- Matplotlib
- Seaborn
tech_stack:
- name: Python
- name: Matplotlib
- name: Seaborn
- name: Scikit-learn
---

## 项目背景  
本项目旨在通过Python的Matplotlib和Seaborn库，实现数据可视化实战。项目涵盖探索性数据分析（EDA）和模型评估两大模块，通过直方图、箱线图、混淆矩阵和ROC曲线等图表，直观展示数据分布特征及模型性能，提升数据分析结果的可读性和说服力。


## 核心实现  

### 1. 探索性数据分析（EDA）  
#### 直方图：年龄分布  
使用Seaborn绘制直方图，展示患者年龄分布：  
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.histplot(data=picu_data, x='age_month', kde=True)
plt.title("年龄分布直方图")
plt.show()
```  

#### 箱线图：不同结局下指标分布  
通过箱线图比较存活与死亡患者的实验室指标差异：  
```python
colname = ['age_month', 'lab_5237_min', 'lab_5227_min', 'lab_5225_range', 'lab_5235_max', 'lab_5257_min']
fig, axs = plt.subplots(3,2, constrained_layout=True, figsize=(10,10))

for i in range(len(colname)):
    sns.boxplot(data=picu_data, x='HOSPITAL_EXPIRE_FLAG', y=colname[i], ax=axs[i//2, i%2])

plt.suptitle("不同结局下各实验室指标分布", fontsize=16)
plt.show()
```  

### 2. 模型评估可视化  
#### 混淆矩阵热力图  
绘制混淆矩阵，评估逻辑回归模型的分类性能：  
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def confusion_matrix_plot(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title('混淆矩阵')
    ax.xaxis.set_ticklabels(['存活(0)', '死亡(1)'])
    ax.yaxis.set_ticklabels(['存活(0)', '死亡(1)'])
    plt.show()

# 调用函数
confusion_matrix_plot(y_test, y_pred_prob, threshold=0.5)
```  

#### ROC曲线  
绘制ROC曲线并计算AUC值：  
```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend()
plt.show()
```  


## 分析结果  

### 1. 探索性分析结果  
![年龄分布直方图](/images/portfolio/python-data-visualization-practice/age_distribution.png)  
直方图显示患者年龄分布呈现双峰特征，集中在低龄和高龄阶段。  

![不同结局下指标分布](/images/portfolio/python-data-visualization-practice/boxplot_distribution.png)  
箱线图表明，死亡患者的部分实验室指标（如lab_5235_max）分布与存活患者存在显著差异，可作为潜在预后因子。  

### 2. 模型评估结果  
![混淆矩阵](/images/portfolio/python-data-visualization-practice/confusion_matrix.png)  
混淆矩阵显示模型对存活患者的预测准确率较高，但对死亡患者的漏诊率略高，需进一步优化。  

![ROC曲线](/images/portfolio/python-data-visualization-practice/roc_curve.png)  
ROC曲线AUC值为0.85，表明模型具有较好的分类性能，能够有效区分存活与死亡患者。  

