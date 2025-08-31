import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import lightgbm as lgb
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DiseasePredictionSystem:
    def __init__(self, data_paths):
        """初始化疾病预测系统，传入数据路径"""
        self.data_paths = data_paths
        
        # 扩展模型库：融合多个优势模型
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True),
            'MLP': MLPClassifier(random_state=42, max_iter=1000),
            'KNeighbors': KNeighborsClassifier(),
            'ExtraTrees': ExtraTreesClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        self.best_models = {}
        self.scalers = {}
        self.results = {}
        self.feature_importances = {}
        self.label_encoders = {}
        self.training_history = {}
        self.shap_values = {}

    def load_and_preprocess(self):
        """加载并预处理所有疾病数据"""
        print("=== 数据预处理 ===")
        self.stroke_data = self._preprocess_stroke()
        self.heart_data = self._preprocess_heart()
        self.cirrhosis_data = self._preprocess_cirrhosis()
        print("数据预处理完成\n")

    def _preprocess_stroke(self):
        """预处理中风数据"""
        df = pd.read_csv(self.data_paths['stroke'])
        print(f"中风数据加载完成，共{len(df)}条记录")
        
        # 处理缺失值：数值型用中位数，分类特征用众数
        df['体重指数'] = df['体重指数'].replace("未知体重指数", np.nan)
        df['体重指数'] = pd.to_numeric(df['体重指数'])
        
        numeric_cols = ['年龄', '平均葡萄糖水平', '体重指数']
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"填充{col}缺失值 {df[col].isnull().sum()}个")
        
        categorical_cols = [
            '性别', '是否高血压', '是否心脏病', '是否结过婚',
            '工作类型', '居住类型', '吸烟状态', '是否中风'
        ]
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"填充{col}缺失值 {df[col].isnull().sum()}个")
        
        # 编码分类变量
        binary_cols = ['是否高血压', '是否心脏病', '是否结过婚', '是否中风']
        for col in binary_cols:
            df[col] = df[col].map({'是': 1, '否': 0})
        
        # 保存标签编码器
        le = LabelEncoder()
        df['性别_编码'] = le.fit_transform(df['性别'])
        self.label_encoders['stroke_gender'] = le
        
        le = LabelEncoder()
        df['工作类型_编码'] = le.fit_transform(df['工作类型'])
        self.label_encoders['stroke_work'] = le
        
        le = LabelEncoder()
        df['居住类型_编码'] = le.fit_transform(df['居住类型'])
        self.label_encoders['stroke_residence'] = le
        
        le = LabelEncoder()
        df['吸烟状态_编码'] = le.fit_transform(df['吸烟状态'])
        self.label_encoders['stroke_smoking'] = le
        
        # 选择特征
        features = [
            '性别_编码', '年龄', '是否高血压', '是否心脏病', '是否结过婚',
            '工作类型_编码', '居住类型_编码', '平均葡萄糖水平', '体重指数', '吸烟状态_编码'
        ]
        X = df[features]
        y = df['是否中风']
        
        # 处理类别不平衡
        print(f"中风类别分布：\n{y.value_counts()}")
        minority_ratio = y.mean()
        
        if minority_ratio < 0.1:  # 如果少数类占比低于 10%
            print("检测到严重数据不平衡，应用 SMOTE 过采样...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"SMOTE 后类别分布：\n{y_resampled.value_counts()}")
            return {
                'X': X_resampled, 'y': y_resampled, 'feature_names': features,
                'original_X': X, 'original_y': y
            }
        elif minority_ratio < 0.2:  # 如果少数类占比低于 20%
            print("检测到中度数据不平衡，将在模型训练中使用 class_weight 参数...")
            return {
                'X': X, 'y': y, 'feature_names': features, 'needs_class_weight': True
            }
        else:
            return {
                'X': X, 'y': y, 'feature_names': features, 'needs_class_weight': False
            }

    def _preprocess_heart(self):
        """预处理心脏病数据"""
        df = pd.read_csv(self.data_paths['heart'])
        print(f"心脏病数据加载完成，共{len(df)}条记录")
        
        # 处理缺失值
        numeric_cols = ['年龄', '静息血压', '血清胆固醇', '最大心率', '抑郁症测量的数值']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
            print(f"填充{col}缺失值 {df[col].isnull().sum()}个")
        
        categorical_cols = [
            '性别', '胸痛型', '空腹血糖是否大于 120 毫克/分升',
            '静息心电图结果', '运动是否诱发心绞痛', '峰值运动ST-T 波异常段的斜率', '是否有心脏病'
        ]
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"填充{col}缺失值 {df[col].isnull().sum()}个")
        
        # 编码分类变量
        binary_cols = ['空腹血糖是否大于 120 毫克/分升', '运动是否诱发心绞痛', '是否有心脏病']
        for col in binary_cols:
            df[col] = df[col].map({'是': 1, '否': 0})
        
        # 保存标签编码器
        le = LabelEncoder()
        df['性别_编码'] = le.fit_transform(df['性别'])
        self.label_encoders['heart_gender'] = le
        
        le = LabelEncoder()
        df['胸痛型_编码'] = le.fit_transform(df['胸痛型'])
        self.label_encoders['heart_chest_pain'] = le
        
        le = LabelEncoder()
        df['静息心电图结果_编码'] = le.fit_transform(df['静息心电图结果'])
        self.label_encoders['heart_ecg'] = le
        
        le = LabelEncoder()
        df['峰值运动 ST-T 波异常段的斜率_编码'] = le.fit_transform(df['峰值运动ST-T 波异常段的斜率'])
        self.label_encoders['heart_st_slope'] = le
        
        # 选择特征
        features = [
            '年龄', '性别_编码', '胸痛型_编码', '静息血压', '血清胆固醇',
            '空腹血糖是否大于 120 毫克/分升', '静息心电图结果_编码', '最大心率',
            '运动是否诱发心绞痛', '抑郁症测量的数值', '峰值运动 ST-T波异常段的斜率_编码'
        ]
        X = df[features]
        y = df['是否有心脏病']
        
        print(f"心脏病类别分布：\n{y.value_counts()}")
        
        # 检查是否需要类别权重
        minority_ratio = y.mean()
        needs_class_weight = minority_ratio < 0.2
        
        return {
            'X': X, 'y': y, 'feature_names': features, 'needs_class_weight': needs_class_weight
        }

    def _preprocess_cirrhosis(self):
        """预处理肝硬化数据"""
        df = pd.read_csv(self.data_paths['cirrhosis'])
        print(f"肝硬化数据加载完成，共{len(df)}条记录")
        
        # 定义目标变量
        df['是否严重肝硬化'] = (df['疾病的组织学阶段(1、2、3 或 4)'] >= 3).astype(int)
        print(f"肝硬化严重程度分布：\n{df['是否严重肝硬化'].value_counts()}")
        
        # 处理缺失值
        numeric_cols = [
            '年龄(年)', '血清胆红素(毫克/分升)', '血清胆固醇(毫克/分升)', '血蛋白(克/分升)',
            '尿铜(微克/天)', '碱性磷酸酶(单位/升)', 'SGOT(单位/毫升)',
            '甘油三酯(毫克/分升)', '血小板(毫克/升)', '凝血酶原时间(s)',
            '疾病的组织学阶段(1、2、3 或 4)'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
            print(f"填充{col}缺失值 {df[col].isnull().sum()}个")
        
        categorical_cols = [
            '性别', '药物类型', '是否存在腹水', '是否存在肝肿大',
            '是否存在 Spiders', '是否存在水肿'
        ]
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"填充{col}缺失值 {df[col].isnull().sum()}个")
        
        # 编码分类变量
        for col in ['是否存在腹水', '是否存在肝肿大', '是否存在 Spiders']:
            df[col + '_编码'] = df[col].map({'是': 1, '否': 0})
        
        # 保存标签编码器
        le = LabelEncoder()
        df['性别_编码'] = le.fit_transform(df['性别'])
        self.label_encoders['cirrhosis_gender'] = le
        
        le = LabelEncoder()
        df['药物类型_编码'] = le.fit_transform(df['药物类型'])
        self.label_encoders['cirrhosis_drug'] = le
        
        le = LabelEncoder()
        df['是否存在水肿_编码'] = le.fit_transform(df['是否存在水肿'])
        self.label_encoders['cirrhosis_edema'] = le
        
        # 选择特征
        features = [
            '年龄(年)', '性别_编码', '药物类型_编码', '是否存在腹水_编码',
            '是否存在肝肿大_编码', '是否存在 Spiders_编码', '是否存在水肿_编码',
            '血清胆红素(毫克/分升)', '血清胆固醇(毫克/分升)', '血小板(毫克/升)',
            '疾病的组织学阶段(1、2、3 或 4)'
        ]
        X = df[features]
        y = df['是否严重肝硬化']
        
        print(f"严重肝硬化类别分布：\n{y.value_counts()}")
        
        # 检查是否需要类别权重
        minority_ratio = y.mean()
        needs_class_weight = minority_ratio < 0.2
        
        return {
            'X': X, 'y': y, 'feature_names': features, 'needs_class_weight': needs_class_weight
        }

    def train_and_evaluate(self):
        """训练并评估所有模型"""
        diseases = [
            ('stroke', '中风', self.stroke_data),
            ('heart', '心脏病', self.heart_data),
            ('cirrhosis', '肝硬化', self.cirrhosis_data)
        ]
        
        for data_key, disease_name, data in diseases:
            print(f"\n=== {disease_name}预测模型 ===")
            X, y = data['X'], data['y']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if data_key != 'stroke' else None
            )
            
            # 数据标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[data_key] = scaler
            
            # 初始化训练历史
            self.training_history[data_key] = {}
            
            # 训练并评估所有模型
            eval_results = {}
            
            for model_name, model in tqdm(self.models.items(), desc=f"训练{disease_name}模型"):
                print(f"\n训练{model_name}...")
                
                try:
                    # 处理类别不平衡
                    if (data.get('needs_class_weight', False) and
                            hasattr(model, 'class_weight')):
                        print(f"应用 class_weight='balanced'处理类别不平衡...")
                        model = model.__class__(**{
                            **model.get_params(),
                            'class_weight': 'balanced'
                        })
                    
                    # 特殊处理支持训练历史的模型
                    history = None
                    
                    if model_name == 'MLP':
                        # MLP 可以记录训练损失
                        model.fit(X_train_scaled, y_train)
                        history = model.loss_curve_
                        self.training_history[data_key][model_name] = {
                            'train_loss': history
                        }
                    
                    elif model_name == 'XGBoost':
                        # XGBoost 可以使用 eval_set 记录评估结果
                        try:
                            eval_set = [(X_train, y_train), (X_test, y_test)]
                            model.fit(
                                X_train, y_train,
                                eval_set=eval_set,
                                early_stopping_rounds=50,
                                verbose=False
                            )
                            history = model.evals_result()
                            self.training_history[data_key][model_name] = {
                                'train_logloss': history['validation_0']['logloss'],
                                'val_logloss': history['validation_1']['logloss']
                            }
                        except TypeError:
                            # 不支持 early_stopping_rounds 参数时的处理
                            model.fit(X_train, y_train)
                    
                    elif model_name == 'LightGBM':
                        # LightGBM 也支持训练历史
                        try:
                            eval_set = [(X_train, y_train), (X_test, y_test)]
                            model.fit(
                                X_train, y_train,
                                eval_set=eval_set,
                                early_stopping_rounds=50,
                                verbose=False
                            )
                            history = model.evals_result_
                            self.training_history[data_key][model_name] = {
                                'train_logloss': history['training']['binary_logloss'],
                                'val_logloss': history['valid_1']['binary_logloss']
                            }
                        except TypeError:
                            # 不支持 early_stopping_rounds 参数时的处理
                            model.fit(X_train, y_train)
                    
                    elif model_name in ['GradientBoosting', 'AdaBoost']:
                        # 梯度提升模型可以通过staged_predict_proba获取训练过程
                        n_estimators = model.n_estimators
                        train_scores = []
                        val_scores = []
                        
                        # 为支持训练进度显示，使用 tqdm 包装训练过程
                        model.fit(X_train, y_train)
                        
                        # 计算每个阶段的性能
                        if hasattr(model, 'staged_predict_proba'):
                            for i, y_pred in enumerate(model.staged_predict_proba(X_train)):
                                train_scores.append(roc_auc_score(y_train, y_pred[:, 1]))
                            
                            for i, y_pred in enumerate(model.staged_predict_proba(X_test)):
                                val_scores.append(roc_auc_score(y_test, y_pred[:, 1]))
                            
                            self.training_history[data_key][model_name] = {
                                'train_auc': train_scores,
                                'val_auc': val_scores
                            }
                    
                    elif model_name in [
                        'LogisticRegression', 'SVM', 'KNeighbors',
                        'RandomForest', 'ExtraTrees'
                    ]:
                        # 为其他模型添加训练历史记录
                        # 这些模型没有内置的训练过程跟踪，所以我们手动创建一些数据点
                        if model_name in ['SVM', 'KNeighbors', 'MLP']:
                            model.fit(X_train_scaled, y_train)
                        else:
                            model.fit(X_train, y_train)
                        
                        # 模拟训练过程：在训练集和测试集上的性能
                        if model_name in ['SVM', 'KNeighbors', 'MLP']:
                            y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
                            y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
                        else:
                            y_train_pred = model.predict_proba(X_train)[:, 1]
                            y_test_pred = model.predict_proba(X_test)[:, 1]
                        
                        self.training_history[data_key][model_name] = {
                            'train_auc': [roc_auc_score(y_train, y_train_pred)],
                            'val_auc': [roc_auc_score(y_test, y_test_pred)]
                        }
                    
                    else:
                        # 其他模型正常训练
                        if model_name in ['SVM', 'KNeighbors', 'MLP']:
                            model.fit(X_train_scaled, y_train)
                        else:
                            model.fit(X_train, y_train)
                    
                    # 预测
                    if model_name in ['SVM', 'KNeighbors', 'MLP']:
                        y_pred = model.predict(X_test_scaled)
                        y_prob = (model.predict_proba(X_test_scaled)[:, 1]
                                  if hasattr(model, 'predict_proba') else None)
                    else:
                        y_pred = model.predict(X_test)
                        y_prob = (model.predict_proba(X_test)[:, 1]
                                  if hasattr(model, 'predict_proba') else None)
                    
                    # 计算评估指标
                    if y_prob is not None:
                        auc = roc_auc_score(y_test, y_prob)
                    else:
                        auc = np.nan
                    
                    eval_results[model_name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'auc': auc,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_prob': y_prob,
                        'model': model
                    }
                    
                    print(f"准确率: {eval_results[model_name]['accuracy']:.4f}")
                    if not np.isnan(auc):
                        print(f"AUC: {eval_results[model_name]['auc']:.4f}")
                    else:
                        print("AUC: 不支持概率输出")
                    
                    # 保存分类报告
                    report = classification_report(
                        y_test, y_pred, target_names=['阴性', '阳性']
                    )
                    eval_results[model_name]['report'] = report
                
                except Exception as e:
                    print(f"模型 {model_name} 训练失败: {e}")
                    eval_results[model_name] = None
            
            # 可视化训练过程
            self._visualize_training_history(data_key, disease_name)
            
            # 选择最佳模型（基于 AUC）
            valid_models = {
                k: v for k, v in eval_results.items()
                if v is not None and not np.isnan(v['auc'])
            }
            
            if not valid_models:
                print(f"未找到支持 AUC 评估的有效模型！")
                continue
            
            best_model_name = max(
                valid_models,
                key=lambda k: valid_models[k]['auc']
            )
            self.best_models[data_key] = valid_models[best_model_name]
            self.results[data_key] = eval_results
            
            print(f"\n最佳模型: {best_model_name} "
                  f"(AUC: {valid_models[best_model_name]['auc']:.4f})")
            print(f"分类报告:\n{valid_models[best_model_name]['report']}")
            
            # 分析与可视化
            self._plot_roc_curve(valid_models[best_model_name], disease_name)
            self._plot_confusion_matrix(valid_models[best_model_name], disease_name)
            self._plot_prediction_distribution(valid_models[best_model_name], disease_name)
            self._analyze_feature_importance(
                X, y, disease_name, best_model_name,
                valid_models[best_model_name]['model']
            )
            self._sensitivity_analysis(
                X, y, best_model_name,
                valid_models[best_model_name]['model'], disease_name, data_key
            )
            self._improve_model(
                X, y, disease_name, best_model_name, data_key
            )
            self._explain_model(
                X, y, best_model_name,
                valid_models[best_model_name]['model'], disease_name, data_key
            )
            
            print("-" * 70)

    def _visualize_training_history(self, data_key, disease_name):
        """可视化模型训练历史"""
        history = self.training_history[data_key]
        
        if not history:
            print(f"没有可用的训练历史数据用于{disease_name}模型")
            return
        
        plt.figure(figsize=(12, 8))
        plt.suptitle(f"{disease_name}模型训练过程")
        
        # 统计有训练历史的模型数量
        model_count = len(history)
        cols = 2
        rows = (model_count + cols - 1) // cols
        
        for i, (model_name, hist) in enumerate(history.items(), 1):
            plt.subplot(rows, cols, i)
            
            if 'train_loss' in hist:
                plt.plot(hist['train_loss'], label='训练损失')
                plt.ylabel('损失')
                plt.title(f'{model_name}训练损失')
            
            elif 'train_logloss' in hist:
                plt.plot(hist['train_logloss'], label='训练 LogLoss')
                plt.plot(hist['val_logloss'], label='验证 LogLoss')
                plt.ylabel('LogLoss')
                plt.title(f'{model_name}训练过程')
            
            elif 'train_auc' in hist:
                # 为所有模型添加训练历史可视化
                if len(hist['train_auc']) > 1:  # 多个训练点
                    plt.plot(hist['train_auc'], label='训练 AUC')
                    plt.plot(hist['val_auc'], label='验证 AUC')
                    plt.xlabel('迭代次数')
                else:  # 只有一个训练点的模型
                    plt.bar(
                        ['训练', '验证'],
                        [hist['train_auc'][0], hist['val_auc'][0]],
                        alpha=0.7
                    )
                    plt.ylim(0, 1.1)
                    plt.ylabel('AUC')
                    plt.title(f'{model_name}模型性能')
            
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 保存图片
        save_path = r"F:\亚太杯\可视化"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        save_file = os.path.join(save_path, f"{disease_name}_训练历史.png")
        plt.savefig(save_file)
        print(f"训练历史图片已保存到: {save_file}")
        plt.show()

    def _analyze_feature_importance(self, X, y, disease_name, model_name, model):
        """分析特征重要性"""
        print(f"\n{disease_name}特征重要性分析")
        
        # 仅对树模型和 XGBoost/LightGBM 分析特征重要性
        if model_name in [
            'RandomForest', 'GradientBoosting', 'ExtraTrees',
            'AdaBoost', 'DecisionTree', 'XGBoost', 'LightGBM'
        ]:
            feature_importance = pd.DataFrame({
                '特征': X.columns,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            self.feature_importances[disease_name] = feature_importance
            
            print("\nTop 10 重要特征:")
            print(feature_importance.head(10))
            
            # 可视化特征重要性
            plt.figure(figsize=(10, 6))
            sns.barplot(x='重要性', y='特征', data=feature_importance.head(10))
            plt.title(f'{disease_name}特征重要性（{model_name}）')
            plt.tight_layout()
            
            # 保存图片
            save_path = r"F:\亚太杯\可视化"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            save_file = os.path.join(
                save_path,
                f"{disease_name}_{model_name}_特征重要性.png"
            )
            plt.savefig(save_file)
            print(f"特征重要性图片已保存到: {save_file}")
            plt.show()

    def _sensitivity_analysis(self, X, y, model_name, model, disease_name, data_key):
        """进行灵敏度分析和交叉验证"""
        print(f"\n{disease_name}模型灵敏度分析")
        
        # 交叉验证评估稳定性
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"交叉验证 AUC: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        # 分析不同特征阈值下的性能变化
        if hasattr(model, 'predict_proba'):
            thresholds = np.linspace(0.1, 0.9, 9)
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for threshold in thresholds:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                if model_name in ['SVM', 'KNeighbors', 'MLP']:
                    # 使用 data_key 而非 disease_name
                    model.fit(self.scalers[data_key].transform(X_train), y_train)
                    y_prob = model.predict_proba(
                        self.scalers[data_key].transform(X_test)
                    )[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_prob = model.predict_proba(X_test)[:, 1]
                
                y_pred_adjusted = (y_prob >= threshold).astype(int)
                precision = precision_score(y_test, y_pred_adjusted)
                recall = recall_score(y_test, y_pred_adjusted)
                f1 = f1_score(y_test, y_pred_adjusted)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                
                print(f"阈值={threshold:.1f}: "
                      f"精确率={precision:.4f}, "
                      f"召回率={recall:.4f}, "
                      f"F1={f1:.4f}")
            
            # 绘制阈值-性能曲线
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precision_scores, 'o-', label='精确率')
            plt.plot(thresholds, recall_scores, 's-', label='召回率')
            plt.plot(thresholds, f1_scores, 'd-', label='F1 分数')
            plt.xlabel('分类阈值')
            plt.ylabel('分数')
            plt.title(f'{disease_name}模型性能随分类阈值变化')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # 特征扰动测试
        print("\n特征扰动测试:")
        
        if model_name not in ['SVM', 'KNeighbors']:
            # 选择 Top 3 重要特征进行扰动测试
            if disease_name in self.feature_importances:
                top_features = self.feature_importances[disease_name]['特征'][:3].tolist()
                X_sample = X.sample(50, random_state=42)  # 选择部分样本进行测试
                
                for feature in top_features:
                    print(f"\n对特征 '{feature}' 进行扰动测试:")
                    
                    original_predictions = model.predict_proba(X_sample)[:, 1]
                    
                    # 增加微小扰动
                    X_perturbed_up = X_sample.copy()
                    X_perturbed_up[feature] = X_perturbed_up[feature] * 1.05  # 增加 5%
                    perturbed_predictions_up = model.predict_proba(X_perturbed_up)[:, 1]
                    
                    # 减少微小扰动
                    X_perturbed_down = X_sample.copy()
                    X_perturbed_down[feature] = X_perturbed_down[feature] * 0.95  # 减少 5%
                    perturbed_predictions_down = model.predict_proba(X_perturbed_down)[:, 1]
                    
                    # 计算平均变化
                    avg_change_up = np.mean(
                        np.abs(perturbed_predictions_up - original_predictions)
                    )
                    avg_change_down = np.mean(
                        np.abs(perturbed_predictions_down - original_predictions)
                    )
                    
                    print(f"增加 5%: 预测概率平均变化 = {avg_change_up:.4f}")
                    print(f"减少 5%: 预测概率平均变化 = {avg_change_down:.4f}")

    def _improve_model(self, X, y, disease_name, best_model_name, data_key):
        """超参数调优和特征选择"""
        print(f"\n{disease_name}模型改进（{best_model_name}）")
        
        # 根据最佳模型类型设置调优参数
        # 扩展所有模型的超参数网格
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': [None, 'balanced']
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            },
            'ExtraTrees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': [None, 'balanced']
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'MLP': {
                'hidden_layer_sizes': [(100,), (100, 50), (50, 50)],
                'learning_rate_init': [0.001, 0.01],
                'alpha': [0.0001, 0.001]
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale']
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        }
        
        # 获取对应模型的参数网格
        param_grid = param_grids.get(best_model_name, {})
        
        # 获取基础模型
        base_model = self.models[best_model_name]
        
        # 网格搜索调优
        print("开始超参数网格搜索...")
        
        # 针对 SVM 单独设置优化参数
        if best_model_name == 'SVM':
            # 启用缓存，并增加并行数
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            # 训练前设置缓存（仅对 SVM 有效）
            base_model.set_params(cache_size=500)  # 500MB 缓存
        else:
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
            )
        
        grid_search.fit(X, y)
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"调优后 AUC: {grid_search.best_score_:.4f}")
        
        # 特征选择
        print("开始特征选择...")
        selector = SelectKBest(f_classif, k=min(10, X.shape[1]))  # 最多选择 10 个特征
        X_selected = selector.fit_transform(X, y)
        
        # 获取被选中的特征
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask]
        print(f"选中的特征: {', '.join(selected_features)}")
        
        # 用选择的特征重新训练模型
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        improved_auc = roc_auc_score(y_test, y_prob)
        
        print(f"特征选择后 AUC: {improved_auc:.4f}")
        
        # 保存改进后的模型
        self.best_models[data_key]['improved_model'] = best_model
        self.best_models[data_key]['selected_features'] = selected_features
        self.best_models[data_key]['improved_auc'] = improved_auc

    def _explain_model(self, X, y, model_name, model, disease_name, data_key):
        """使用 SHAP 值解释模型预测逻辑"""
        print(f"\n{disease_name}模型解释性分析")
        
        # 仅对支持 SHAP 的模型进行解释
        if model_name in [
            'RandomForest', 'GradientBoosting', 'ExtraTrees',
            'AdaBoost', 'XGBoost', 'LightGBM'
        ]:
            try:
                # 使用 SHAP 解释模型
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # 保存 SHAP 值供后续使用
                self.shap_values[data_key] = shap_values
                
                # 可视化 SHAP 摘要图
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X, show=False)
                plt.title(f'{disease_name}模型 SHAP 特征重要性')
                plt.tight_layout()
                
                # 保存图片
                save_path = r"F:\亚太杯\可视化"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                save_file = os.path.join(
                    save_path,
                    f"{disease_name}_{model_name}_SHAP_摘要.png"
                )
                plt.savefig(save_file)
                print(f"SHAP 摘要图片已保存到: {save_file}")
                plt.show()
                
                # 可视化单个样本的预测解释
                sample_idx = 0  # 选择第一个样本进行解释
                plt.figure(figsize=(10, 6))
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[sample_idx, :],
                    X.iloc[sample_idx, :],
                    show=False,
                    matplotlib=True
                )
                plt.title(f'{disease_name}模型对样本 {sample_idx} 的预测解释')
                plt.tight_layout()
                
                save_file = os.path.join(
                    save_path,
                    f"{disease_name}_{model_name}_SHAP_样本解释.png"
                )
                plt.savefig(save_file)
                print(f"SHAP 样本解释图片已保存到: {save_file}")
                plt.show()
                
                # 分析特征对预测的影响方向
                print("\n特征对预测的影响方向:")
                for i, feature in enumerate(X.columns):
                    # 计算该特征 SHAP 值与特征值的相关性
                    correlation = np.corrcoef(X[feature], shap_values[:, i])[0, 1]
                    print(f"{feature}: 相关性 = {correlation:.4f}")
            
            except Exception as e:
                print(f"SHAP 解释失败: {e}")

    def _plot_roc_curve(self, result, disease_name):
        """绘制 ROC 曲线"""
        if result['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_prob'])
            auc_score = result['auc']
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title(f'{disease_name} ROC 曲线')
            plt.legend(loc='lower right')
            plt.grid(True)
            
            # 保存图片
            save_path = r"F:\亚太杯\可视化"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            save_file = os.path.join(save_path, f"{disease_name}_ROC曲线.png")
            plt.savefig(save_file)
            print(f"ROC 曲线图片已保存到: {save_file}")
            plt.show()

    def _plot_confusion_matrix(self, result, disease_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['阴性', '阳性'],
            yticklabels=['阴性', '阳性']
        )
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{disease_name}混淆矩阵')
        
        # 保存图片
        save_path = r"F:\亚太杯\可视化"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        save_file = os.path.join(save_path, f"{disease_name}_混淆矩阵.png")
        plt.savefig(save_file)
        print(f"混淆矩阵图片已保存到: {save_file}")
        plt.show()

    def _plot_prediction_distribution(self, result, disease_name):
        """绘制预测概率分布"""
        if result['y_prob'] is not None:
            plt.figure(figsize=(10, 6))
            
            # 按真实类别分组绘制预测概率分布
            for label in [0, 1]:
                mask = result['y_test'] == label
                plt.hist(
                    result['y_prob'][mask],
                    bins=30, alpha=0.7,
                    label=f'真实类别: {label}',
                    density=True
                )
            
            plt.xlabel('预测概率')
            plt.ylabel('密度')
            plt.title(f'{disease_name}预测概率分布')
            plt.legend()
            plt.grid(True)
            
            # 保存图片
            save_path = r"F:\亚太杯\可视化"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            save_file = os.path.join(
                save_path,
                f"{disease_name}_预测概率分布.png"
            )
            plt.savefig(save_file)
            print(f"预测概率分布图片已保存到: {save_file}")
            plt.show()

    def generate_report(self):
        """生成综合评估报告"""
        print("\n=== 综合评估报告 ===")
        
        for data_key, disease_name in [
            ('stroke', '中风'),
            ('heart', '心脏病'),
            ('cirrhosis', '肝硬化')
        ]:
            if data_key in self.best_models:
                result = self.best_models[data_key]
                print(f"\n{disease_name}预测模型:")
                print(f"最佳模型: {type(result['model']).__name__}")
                print(f"准确率: {result['accuracy']:.4f}")
                print(f"AUC: {result['auc']:.4f}")
                print(f"F1 分数: {result['f1']:.4f}")
                
                if 'improved_auc' in result:
                    print(f"改进后 AUC: {result['improved_auc']:.4f}")
                
                # 打印重要特征
                if disease_name in self.feature_importances:
                    print("Top 3 重要特征:")
                    top_features = self.feature_importances[disease_name].head(3)
                    for _, row in top_features.iterrows():
                        print(f"  {row['特征']}: {row['重要性']:.4f}")

    def predict_new_sample(self, disease_type, features):
        """预测新样本"""
        if disease_type not in self.best_models:
            print(f"没有找到{disease_type}的模型")
            return None
        
        model_info = self.best_models[disease_type]
        model = model_info['model']
        
        # 确保特征顺序正确
        if 'selected_features' in model_info:
            # 使用改进后的特征集
            selected_features = model_info['selected_features']
            features = features[selected_features]
        
        # 预测
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba([features])[0, 1]
            prediction = model.predict([features])[0]
            
            print(f"预测结果: {'阳性' if prediction == 1 else '阴性'}")
            print(f"阳性概率: {probability:.4f}")
            
            return prediction, probability
        else:
            prediction = model.predict([features])[0]
            print(f"预测结果: {'阳性' if prediction == 1 else '阴性'}")
            
            return prediction, None


if __name__ == "__main__":
    # 数据路径
    data_paths = {
        'stroke': r'F:\亚太杯\Stroke.csv',
        'heart': r'F:\亚太杯\Heart.csv',
        'cirrhosis': r'F:\亚太杯\Cirrhosis.csv'
    }
    
    # 创建预测系统
    system = DiseasePredictionSystem(data_paths)
    
    # 加载和预处理数据
    system.load_and_preprocess()
    
    # 训练和评估模型
    system.train_and_evaluate()
    
    # 生成报告
    system.generate_report()
    