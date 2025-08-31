# Multi-Model Fusion and Bayesian Network-Driven Prediction of Chronic Diseases and Their Comorbidity Analysis

[![APMCM](https://img.shields.io/badge/Competition-APMCM-orange)](https://www.apmcm.org/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-green)](https://scikit-learn.org/)

This project, based on medical data for three highly prevalent chronic diseases—heart disease, stroke, and cirrhosis—constructs an end-to-end scientific solution. This solution encompasses data governance, precise prediction, comorbidity mechanism analysis, and the translation of findings into prevention and control strategies.

## 📁 Project Structure

```bash
Project_Root/
│
├── 📁 Problem/                         # Competition Problem & Raw Data
│   ├── 📊 cirrhosis.csv               # Cirrhosis Dataset
│   ├── 📊 heart.csv                   # Heart Disease Dataset
│   ├── 📊 stroke.csv                  # Stroke Dataset
│   ├── 📄 Appendix_Dataset_Description.docx # Field description document for the original datasets
│   └── 📄 Disease_Prediction_and_Big_Data_Analysis.pdf # Competition problem document
│
├── 📄 Paper.pdf                       # Complete solution paper, including problem analysis, model establishment, solution process, and conclusions
├── 📊 Data_Analysis_and_Visualization.py  # Script for data preprocessing, descriptive statistics, hypothesis testing, and visualization
├── 🤖 Disease_prediction_model.py         # Script for building, training, and evaluating disease prediction models (MLP, XGBoost, SVM, RF, etc.)
├── 🔗 Disease_analysis.py                 # Script for multi-disease association analysis and comprehensive risk assessment (Bayesian Network, etc.)
├── 📈 Visualizations/                     # Folder containing all generated analysis charts and result images
│   ├── Multi_Disease_Risk_Probability.png
│   ├── Multi_Disease_Age_Distribution.png
│   ├── ... (Other visualizations)
│   └── ...
│
└── 📖 README.md                       # This project description file
```

## 🎯 Project Objectives

This project aims to address the following four core problems:

1.  **Data Governance & Exploratory Data Analysis (EDA)**: Clean heterogeneous medical data, handle missing values, detect anomalies, and reveal key statistical characteristics and distribution patterns for each disease.
2.  **Disease Prediction Model Construction**: Build high-precision machine learning prediction models (MLP, SVM, XGBoost) for heart disease, stroke, and cirrhosis, respectively.
3.  **Comorbidity Mechanism Analysis**: Explore the associations between the three diseases, quantify comorbidity probabilities, and construct a multi-disease association network using a Bayesian Network.
4.  **Public Health Strategy Recommendations**: Formulate a report with early screening and prevention strategy recommendations for high-risk groups based on the analysis results.

## ⚙️ Environment Dependencies

Python 3.7 or higher is recommended. Main dependencies are listed below:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm shap pgmpy
```

*(Note: Added `pgmpy` for Bayesian Network functionality)*

## 🚀 Quick Start

### 1. Prepare Data
Place the three raw data files (`stroke.csv`, `heart.csv`, `cirrhosis.csv`) into the `Problem/` folder within the project directory.

### 2. Run the Analysis Pipeline
It is recommended to execute the scripts in the following order to reproduce all results from the paper:

**a. Data Exploration & Visualization**
```bash
python Data_Analysis_and_Visualization.py
```
*This script performs data preprocessing, generates descriptive statistics tables, and preliminary visualizations.*

**b. Train & Evaluate Prediction Models**
```bash
python Disease_prediction_model.py
```
*This script trains various machine learning models for the three diseases, outputs performance evaluations (AUC, Accuracy, Recall, etc.), and saves the best models.*

**c. Perform Multi-Disease Association Analysis**
```bash
python Disease_analysis.py
```
*This script analyzes common features among diseases, calculates comorbidity probabilities, and generates related comprehensive risk assessment charts.*

### 3. View Results
All generated analysis charts, ROC curves, confusion matrices, etc., will be automatically saved to the `Visualizations/` folder. Final model performance conclusions and comorbidity probability analysis can be found in `Paper.pdf`.

## 📊 Model Performance

| Disease | Best Model | AUC | Accuracy | Recall (Sensitivity) |
| :--- | :--- | :--- | :--- | :--- |
| **Cirrhosis** | MLP | 0.9951 | 0.976 | 0.962 |
| **Heart Disease** | SVM | 0.9443 | 0.880 | 0.870 |
| **Stroke** | XGBoost | **0.9951** | 0.944 | 0.915 |

## 📋 Key Findings

-   **Comorbidity Risk**: The "Heart Disease + Cirrhosis" comorbidity has the highest probability (0.213), which is 20.5 times higher than the probability of having all three diseases simultaneously.
-   **Core Hub Factor**: Cholesterol is a key hub factor for comorbidities. For every 1mmol/L increase, the comorbidity risk increases by 65%.
-   **Age Stratification**: The comorbidity risk in the elderly group (≥60 years) is 2.63 times that of the youth group.
-   **Model Robustness**: All models demonstrated strong robustness (AUC fluctuation <0.018) under feature perturbation tests (±5%).

## 📝 Detailed Description

-   **`Paper.pdf`**: Contains complete project details, including problem restatement, model assumptions, notation, detailed modeling and solution process, result analysis, and final conclusions and recommendation letter.
-   **Python Scripts**: Each script is highly modularized and contains detailed comments for easy understanding and modification.
    -   `Data_Analysis_and_Visualization.py`: Focuses on data cleaning and statistical testing.
    -   `Disease_prediction_model.py`: Implements various machine learning algorithms and a rigorous model evaluation pipeline.
    -   `Disease_analysis.py`: Focuses on multi-disease association analysis and Bayesian network construction.

## 👥 Author

*YANKEESEAN*

## 📄 License

This project is intended for academic research only. Data usage must comply with respective license agreements.

---

# 多模型融合与贝叶斯网络驱动的慢性病预测及其共病分析

[![APMCM](https://img.shields.io/badge/Competition-APMCM-orange)](https://www.apmcm.org/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-green)](https://scikit-learn.org/)

本项目基于心脏病、中风和肝硬化三类高发慢性疾病的医疗数据，构建了一套从数据治理、精准预测到共病机制解析和防控策略转化的全链条科学解决方案。

## 📁 项目结构

```bash
项目根目录/
│
├── 📁 题目/                         # 竞赛题目与原始数据
│   ├── 📊 cirrhosis.csv            # 肝硬化数据集
│   ├── 📊 heart.csv               # 心脏病数据集
│   ├── 📊 stroke.csv              # 中风数据集
│   ├── 📄 附录：数据集说明.docx     # 原始数据集的字段说明文档
│   └── 📄 疾病的预测与大数据分析.pdf # 竞赛题目文档
│
├── 📄 论文.pdf                    # 本项目完整的解题论文，包含问题分析、模型建立、求解过程及结论
├── 📊 Data_Analysis_and_Visualization.py  # 数据预处理、描述性统计、差异性检验及可视化脚本
├── 🤖 Disease_prediction_model.py         # 疾病预测模型构建、训练与评估脚本（MLP, XGBoost, SVM, RF等）
├── 🔗 Disease_analysis.py                 # 多疾病关联分析与综合风险评估脚本（贝叶斯网络等）
├── 📈 可视化/                           # 存放所有生成的分析图表和结果图片的文件夹
│   ├── 多疾病风险概率.png
│   ├── 多疾病年龄分布.png
│   ├── ... (其他可视化图片)
│   └── ...
│
└── 📖 README.md                   # 本项目说明文件
```

## 🎯 项目目标

本项目旨在解决以下四个核心问题：

1.  **数据治理与探索性分析 (EDA)**：对异构医疗数据进行清洗、缺失值处理和异常值检测，揭示各疾病的关键统计特征和分布规律。
2.  **疾病预测模型构建**：为心脏病、中风和肝硬化分别建立高精度的机器学习预测模型（MLP, SVM, XGBoost）。
3.  **共病机制解析**：探索三种疾病之间的关联性，量化共病概率，并使用贝叶斯网络构建多病关联网络。
4.  **公共卫生策略建议**：基于分析结果，形成针对高风险人群的早期筛查和预防策略建议报告。

## ⚙️ 环境依赖

建议使用 Python 3.7 或更高版本。主要依赖库如下：

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm shap
```

## 🚀 快速开始

### 1. 准备数据
将三个原始数据文件 (`stroke.csv`, `heart.csv`, `cirrhosis.csv`) 放入项目目录下的 `数据文件/` 文件夹中。

### 2. 运行分析流程
建议按以下顺序执行脚本，以复现论文中的全部结果：

**a. 数据探索与可视化**
```bash
python Data_Analysis_and_Visualization.py
```
*此脚本将进行数据预处理、生成描述性统计表和初步的可视化图表。*

**b. 训练与评估预测模型**
```bash
python Disease_prediction_model.py
```
*此脚本将训练针对三种疾病的多种机器学习模型，输出性能评估（AUC, 准确率, 召回率等）并保存最佳模型。*

**c. 进行多疾病关联分析**
```bash
python Disease_analysis.py
```
*此脚本将分析疾病间的共同特征，计算共病概率，并生成相关的综合风险评估图表。*

### 3. 查看结果
所有生成的分析图表、ROC曲线、混淆矩阵等将自动保存至 `可视化/` 文件夹。最终的模型性能结论和共病概率分析请参考 `论文.pdf`。

## 📊 模型性能

| 疾病 | 最佳模型 | AUC | 准确率 | 召回率 |
| :--- | :--- | :--- | :--- | :--- |
| **肝硬化** | MLP | 0.9951 | 0.976 | 0.962 |
| **心脏病** | SVM | 0.9443 | 0.880 | 0.870 |
| **中风** | XGBoost | **0.9951** | 0.944 | 0.915 |

## 📋 核心发现

-   **共病风险**：”心脏病+肝硬化”共病概率最高（0.213），是“三病共存”概率的20.5倍。
-   **核心枢纽**：胆固醇是共病的关键枢纽因子，每升高1mmol/L，共病风险增加65%。
-   **年龄分层**：老年组（≥60岁）的共病风险是青年组的2.63倍。
-   **模型鲁棒性**：经特征扰动测试（±5%），所有模型均表现出强鲁棒性（AUC波动<0.018）。

## 📝 详细说明

-   **`论文.pdf`**：包含了项目的完整细节，包括问题重述、模型假设、符号说明、详细的建模求解过程、结果分析以及最终的结论和建议信。
-   **Python脚本**：每个脚本都高度模块化，包含详细的注释，便于理解和修改。
    -   `Data_Analysis_and_Visualization.py`: 专注于数据清洗和统计检验。
    -   `Disease_prediction_model.py`: 实现了多种机器学习算法和严格的模型评估流程。
    -   `Disease_analysis.py`: 侧重于多疾病之间的关联性分析和贝叶斯网络构建。

## 👥 作者

*YANKEESEAN*

## 📄 许可证

本项目仅用于学术研究。数据使用请遵循相应的许可协议。

---
