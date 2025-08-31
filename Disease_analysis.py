import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义特征名的中文映射
feature_chinese_mapping = {
    'gender_encoded': '性别',
    'Age': '年龄',
    'age': '年龄',
    'hypertension': '是否有高血压',
    'heart_disease': '是否有心脏病',
    'ever_married_encoded': '是否已婚',
    'work_type_encoded': '工作类型',
    'Residence_type_encoded': '居住类型',
    'avg_glucose_level': '平均葡萄糖水平',
    'bmi': '体重指数',
    'smoking_status_encoded': '吸烟状态',
    'Sex_encoded': '性别',
    'Drug_encoded': '药物类型',
    'Ascites_encoded': '是否存在腹水',
    'Hepatomegaly_encoded': '是否存在肝肿大',
    'Spiders_encoded': '是否存在蜘蛛痣',
    'Edema_encoded': '是否存在水肿',
    'Bilirubin': '血清胆红素',
    'Cholesterol': '血清胆固醇',
    'Albumin': '白蛋白',
    'Copper': '尿铜',
    'Alk_Phos': '碱性磷酸酶',
    'SGOT': '谷草转氨酶',
    'Tryglicerides': '甘油三酯',
    'Platelets': '每立方血小板',
    'Prothrombin': '凝血酶原时间',
    'Stage': '疾病的组织学阶段',
    'ChestPainType_encoded': '胸痛型',
    'RestingBP': '静息血压',
    'FastingBS': '空腹血糖',
    'RestingECG_encoded': '静息心电图结果',
    'MaxHR': '最大心率',
    'ExerciseAngina_encoded': '运动性心绞痛',
    'Oldpeak': '运动后 ST 段压低',
    'ST_Slope_encoded': 'ST 段斜率'
}


class DiseaseAnalyzer:
    def __init__(self):
        # 创建保存图片的文件夹
        self.save_folder = "E:\\桌面\\可视化"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("=== 数据加载和预处理 ===")
        
        # 加载数据
        try:
            self.stroke_data = pd.read_csv('stroke.csv')
            self.cirrhosis_data = pd.read_csv('cirrhosis.csv')
            self.heart_data = pd.read_csv('heart.csv')
            print("数据加载成功")
        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
            return False
        
        # 预处理中风数据
        self.stroke_processed = self.preprocess_stroke_data()
        
        # 预处理肝硬化数据
        self.cirrhosis_processed = self.preprocess_cirrhosis_data()
        
        # 预处理心脏病数据
        self.heart_processed = self.preprocess_heart_data()
        
        print("数据预处理完成")
        return True

    def preprocess_stroke_data(self):
        """预处理中风数据"""
        data = self.stroke_data.copy()
        
        # 处理缺失值
        data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
        data['bmi'] = data['bmi'].fillna(data['bmi'].median())
        
        # 编码分类变量
        le_gender = LabelEncoder()
        data['gender_encoded'] = le_gender.fit_transform(data['gender'])
        
        le_married = LabelEncoder()
        data['ever_married_encoded'] = le_married.fit_transform(data['ever_married'])
        
        le_work = LabelEncoder()
        data['work_type_encoded'] = le_work.fit_transform(data['work_type'])
        
        le_residence = LabelEncoder()
        data['Residence_type_encoded'] = le_residence.fit_transform(data['Residence_type'])
        
        le_smoking = LabelEncoder()
        data['smoking_status_encoded'] = le_smoking.fit_transform(data['smoking_status'])
        
        # 选择特征
        features = [
            'gender_encoded', 'age', 'hypertension', 'heart_disease',
            'ever_married_encoded', 'work_type_encoded', 'Residence_type_encoded',
            'avg_glucose_level', 'bmi', 'smoking_status_encoded'
        ]
        X = data[features]
        y = data['stroke']
        
        return {'X': X, 'y': y, 'feature_names': features}

    def preprocess_cirrhosis_data(self):
        """预处理肝硬化数据"""
        data = self.cirrhosis_data.copy()
        
        # 将年龄从以天为单位转换为以年为单位
        data['Age'] = data['Age'] / 365
        
        # 创建二分类目标变量（D 表示死亡，其他表示存活）
        data['cirrhosis_death'] = (data['Status'] == 'D').astype(int)
        
        # 打印缺失值统计信息，用于调试
        print("\n肝硬化数据缺失值统计:")
        print(data.isnull().sum())
        
        # 处理数值列缺失值
        numeric_columns = [
            'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
            'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage'
        ]
        
        # 确保所有数值列都在数据中
        available_numeric = [col for col in numeric_columns if col in data.columns]
        
        # 使用中位数填充数值列缺失值
        imputer = SimpleImputer(strategy='median')
        data[available_numeric] = imputer.fit_transform(data[available_numeric])
        
        # 处理分类列缺失值
        categorical_columns = ['Sex', 'Drug', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
        
        # 为每个分类列创建缺失值指示列，并填充缺失值为最频繁值
        for col in categorical_columns:
            if col in data.columns:
                # 创建缺失值指示列
                data[col + '_missing'] = data[col].isnull().astype(int)
                data[col] = data[col].fillna(data[col].mode()[0])
        
        # 编码分类变量
        self.le_sex = LabelEncoder()  # 保存实例供后续使用
        data['Sex_encoded'] = self.le_sex.fit_transform(data['Sex'])
        
        le_drug = LabelEncoder()
        data['Drug_encoded'] = le_drug.fit_transform(data['Drug'])
        
        # 处理 Y/N 变量
        yn_columns = ['Ascites', 'Hepatomegaly', 'Spiders']
        for col in yn_columns:
            if col in data.columns:
                data[col + '_encoded'] = data[col].map({'Y': 1, 'N': 0})
        
        # 处理 Edema 变量
        edema_mapping = {'N': 0, 'S': 1, 'Y': 2}
        data['Edema_encoded'] = data['Edema'].map(edema_mapping)
        
        # 选择特征
        features = [
            'Age', 'Sex_encoded', 'Drug_encoded', 'Ascites_encoded',
            'Hepatomegaly_encoded', 'Spiders_encoded', 'Edema_encoded',
            'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
            'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage'
        ]
        
        # 确保所有特征都存在
        available_features = [f for f in features if f in data.columns]
        
        X = data[available_features]
        y = data['cirrhosis_death']
        
        # 验证处理后的数据是否还有缺失值
        print("\n预处理后肝硬化数据缺失值统计:")
        missing_values = X.isnull().sum()
        print(missing_values)
        
        # 输出具体包含缺失值的列
        if missing_values.sum() > 0:
            print("\n包含缺失值的列:")
            print(missing_values[missing_values > 0])
        
        # 确保没有缺失值
        assert X.isnull().sum().sum() == 0, f"预处理后的数据仍包含缺失值! 缺失值总数: {missing_values.sum()}"
        
        return {'X': X, 'y': y, 'feature_names': available_features}

    def preprocess_heart_data(self):
        """预处理心脏病数据"""
        data = self.heart_data.copy()
        
        # 编码分类变量
        le_sex = LabelEncoder()
        data['Sex_encoded'] = le_sex.fit_transform(data['Sex'])
        
        le_chest_pain = LabelEncoder()
        data['ChestPainType_encoded'] = le_chest_pain.fit_transform(data['ChestPainType'])
        
        le_resting_ecg = LabelEncoder()
        data['RestingECG_encoded'] = le_resting_ecg.fit_transform(data['RestingECG'])
        
        le_exercise_angina = LabelEncoder()
        data['ExerciseAngina_encoded'] = le_exercise_angina.fit_transform(data['ExerciseAngina'])
        
        le_st_slope = LabelEncoder()
        data['ST_Slope_encoded'] = le_st_slope.fit_transform(data['ST_Slope'])
        
        # 选择特征
        features = [
            'Age', 'Sex_encoded', 'ChestPainType_encoded', 'RestingBP',
            'Cholesterol', 'FastingBS', 'RestingECG_encoded', 'MaxHR',
            'ExerciseAngina_encoded', 'Oldpeak', 'ST_Slope_encoded'
        ]
        X = data[features]
        y = data['HeartDisease']
        
        return {'X': X, 'y': y, 'feature_names': features}

    def multi_disease_analysis(self):
        """多疾病关联分析"""
        print("\n=== 多疾病关联与综合风险评估 ===")
        
        # 创建综合数据集
        # 由于数据来源不同，我们需要创建一个统一的评估框架
        
        # 使用概率方法进行多疾病风险评估
        self.estimate_multi_disease_risk()
        
        # 共同特征分析
        self.analyze_common_features()
        
        # 深入共同特征分析
        self.analyze_deeper_common_features()

    def estimate_multi_disease_risk(self):
        """估计多疾病风险"""
        print("\n--- 多疾病风险概率估计 ---")
        
        # 假设各疾病独立，计算联合概率
        
        # 获取各疾病的患病概率
        stroke_prob = self.stroke_processed['y'].mean()
        heart_prob = self.heart_processed['y'].mean()
        cirrhosis_prob = self.cirrhosis_processed['y'].mean()
        
        print(f"中风患病概率: {stroke_prob:.4f}")
        print(f"心脏病患病概率: {heart_prob:.4f}")
        print(f"肝硬化死亡概率: {cirrhosis_prob:.4f}")
        
        # 计算两种疾病同时发生的概率（假设独立）
        stroke_heart_prob = stroke_prob * heart_prob
        stroke_cirrhosis_prob = stroke_prob * cirrhosis_prob
        heart_cirrhosis_prob = heart_prob * cirrhosis_prob
        
        # 计算三种疾病同时发生的概率
        all_three_prob = stroke_prob * heart_prob * cirrhosis_prob
        
        print(f"\n两种疾病同时发生的概率:")
        print(f"中风+心脏病: {stroke_heart_prob:.6f}")
        print(f"中风+肝硬化: {stroke_cirrhosis_prob:.6f}")
        print(f"心脏病+肝硬化: {heart_cirrhosis_prob:.6f}")
        print(f"\n三种疾病同时发生的概率: {all_three_prob:.8f}")
        
        # 可视化多疾病风险概率
        diseases = ['中风', '心脏病', '肝硬化', '中风+心脏病', '中风+肝硬化', '心脏病+肝硬化', '三种疾病']
        probabilities = [
            stroke_prob, heart_prob, cirrhosis_prob, stroke_heart_prob, stroke_cirrhosis_prob,
            heart_cirrhosis_prob, all_three_prob
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(diseases, probabilities)
        for i, v in enumerate(probabilities):
            plt.text(i, v, f'{v:.6f}', ha='center')
        
        plt.xlabel('疾病组合')
        plt.ylabel('患病概率')
        plt.title('多疾病风险概率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder, '多疾病风险概率.png'))
        plt.show()

    def analyze_common_features(self):
        """分析共同特征"""
        print("\n--- 共同特征分析 ---")
        
        # 年龄分布比较
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(self.stroke_processed['X']['age'], bins=30, alpha=0.7, label='中风')
        plt.xlabel('年龄')
        plt.ylabel('频数')
        plt.title('中风患者年龄分布')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.hist(self.heart_processed['X']['Age'], bins=30, alpha=0.7, label='心脏病', color='orange')
        plt.xlabel('年龄')
        plt.ylabel('频数')
        plt.title('心脏病患者年龄分布')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.hist(self.cirrhosis_processed['X']['Age'], bins=30, alpha=0.7, label='肝硬化', color='green')
        plt.xlabel('年龄')
        plt.ylabel('频数')
        plt.title('肝硬化患者年龄分布')
        plt.legend()
        
        plt.tight_layout()
        # 保存图片
        plt.savefig(os.path.join(self.save_folder, '多疾病年龄分布.png'))
        plt.show()
        
        # 性别分布比较
        plt.figure(figsize=(15, 5))
        
        # 中风数据性别分布
        plt.subplot(1, 3, 1)
        stroke_gender_counts = self.stroke_processed['X']['gender_encoded'].value_counts().sort_index()
        stroke_gender_labels = ['女性', '男性', '其他']
        stroke_gender_labels = stroke_gender_labels[:len(stroke_gender_counts)]
        bars = plt.bar(stroke_gender_labels, stroke_gender_counts)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')
        plt.title('中风患者性别分布')
        
        # 心脏病数据性别分布
        plt.subplot(1, 3, 2)
        heart_gender_counts = self.heart_processed['X']['Sex_encoded'].value_counts().sort_index()
        heart_gender_labels = ['女性', '男性']
        heart_gender_labels = heart_gender_labels[:len(heart_gender_counts)]
        bars = plt.bar(heart_gender_labels, heart_gender_counts)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')
        plt.title('心脏病患者性别分布')
        
        # 肝硬化数据性别分布
        plt.subplot(1, 3, 3)
        cirrhosis_gender_counts = self.cirrhosis_processed['X']['Sex_encoded'].value_counts().sort_index()
        if hasattr(self, 'le_sex') and hasattr(self.le_sex, 'classes_'):
            # 获取 LabelEncoder 的类别映射
            gender_mapping = {v: k for k, v in enumerate(self.le_sex.classes_)}
            # 反转映射并创建标签
            cirrhosis_gender_labels = ['女性' if code == gender_mapping.get('F', 0) else '男性' for code in cirrhosis_gender_counts.index]
        else:
            # 默认情况，以防 LabelEncoder 未正确保存
            cirrhosis_gender_labels = ['女性', '男性'][:len(cirrhosis_gender_counts)]
        
        bars = plt.bar(cirrhosis_gender_labels, cirrhosis_gender_counts)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')
        plt.title('肝硬化患者性别分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder, '多疾病性别分布.png'))
        plt.show()

    def compare_common_numeric_features(self, features):
        """比较共同数值特征在不同疾病中的分布"""
        for feature in features:
            plt.figure(figsize=(15, 5))
            
            # 确保特征名称在不同数据集中的一致性
            feature_mapping = {
                'Age': {'stroke': 'age', 'heart': 'Age', 'cirrhosis': 'Age'},
                'avg_glucose_level': {'stroke': 'avg_glucose_level', 'heart': None, 'cirrhosis': None},
            }
            
            # 绘制各疾病特征分布
            plot_position = 1
            for disease, color, title in zip(
                ['stroke', 'heart', 'cirrhosis'],
                ['blue', 'orange', 'green'],
                ['中风', '心脏病', '肝硬化']
            ):
                disease_feature = feature_mapping[feature][disease]
                if disease_feature and disease_feature in getattr(self, f'{disease}_processed')['X'].columns:
                    plt.subplot(1, 3, plot_position)
                    plt.hist(getattr(self, f'{disease}_processed')['X'][disease_feature], bins=30, alpha=0.7, color=color)
                    plt.title(f'{title}患者{feature_chinese_mapping.get(disease_feature, disease_feature)}分布')
                    plt.xlabel(feature_chinese_mapping.get(disease_feature, disease_feature))
                    plt.ylabel('频数')
                    plot_position += 1
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, f'共同特征_{feature}_分布比较.png'))
            plt.show()

    def analyze_deeper_common_features(self):
        """深入分析共同特征"""
        print("\n--- 深入共同特征分析 ---")
        
        # 找出三种疾病共有的特征
        stroke_features = set(self.stroke_processed['feature_names'])
        heart_features = set(self.heart_processed['feature_names'])
        cirrhosis_features = set(self.cirrhosis_processed['feature_names'])
        
        # 共同特征
        common_features = stroke_features.intersection(heart_features).intersection(cirrhosis_features)
        
        # 中风与心脏病的共同特征
        stroke_heart_common = stroke_features.intersection(heart_features)
        
        # 中风与肝硬化的共同特征
        stroke_cirrhosis_common = stroke_features.intersection(cirrhosis_features)
        
        # 心脏病与肝硬化的共同特征
        heart_cirrhosis_common = heart_features.intersection(cirrhosis_features)
        
        print("\n三种疾病的共同特征:")
        for feature in common_features:
            chinese_name = feature_chinese_mapping.get(feature, feature)
            print(f"- {feature} ({chinese_name})")
        
        print("\n中风与心脏病的共同特征:")
        for feature in stroke_heart_common - common_features:
            chinese_name = feature_chinese_mapping.get(feature, feature)
            print(f"- {feature} ({chinese_name})")
        
        print("\n中风与肝硬化的共同特征:")
        for feature in stroke_cirrhosis_common - common_features:
            chinese_name = feature_chinese_mapping.get(feature, feature)
            print(f"- {feature} ({chinese_name})")
        
        print("\n心脏病与肝硬化的共同特征:")
        for feature in heart_cirrhosis_common - common_features:
            chinese_name = feature_chinese_mapping.get(feature, feature)
            print(f"- {feature} ({chinese_name})")
        
        # 可视化共同特征
        self.visualize_common_features(common_features)
        
        # 分析共同数值特征的分布
        self.analyze_common_numeric_features()
        
        # 明确列出可能的共同特征
        self.identify_potential_common_features()

    def identify_potential_common_features(self):
        """识别潜在的共同特征，处理特征名称不一致的情况"""
        print("\n--- 潜在共同特征分析 ---")
        
        # 定义可能的共同特征及其在不同数据集中的名称
        potential_common = {
            '年龄': {
                'stroke': 'age',
                'heart': 'Age',
                'cirrhosis': 'Age'
            },
            '性别': {
                'stroke': 'gender',
                'heart': 'Sex',
                'cirrhosis': 'Sex'
            },
            '胆固醇': {
                'stroke': None,  # 中风数据集中没有胆固醇
                'heart': 'Cholesterol',
                'cirrhosis': 'Cholesterol'
            },
            '心脏病史': {
                'stroke': 'heart_disease',
                'heart': 'HeartDisease',
                'cirrhosis': None
            }
        }
        
        # 分析每种潜在共同特征
        for feature_name, datasets in potential_common.items():
            present_in = []
            for dataset, col_name in datasets.items():
                if col_name and col_name in getattr(self, f'{dataset}_processed')['X'].columns:
                    present_in.append(dataset)
            if len(present_in) >= 2:
                print(f"\n{feature_name} 存在于: {', '.join([d.title() for d in present_in])}")
                print("数据列名:")
                for dataset in present_in:
                    col_name = datasets[dataset]
                    print(f" - {dataset.title()}: {col_name} ({feature_chinese_mapping.get(col_name, col_name)})")

    def visualize_common_features(self, common_features):
        """可视化共同特征"""
        # 提取数值型共同特征
        numeric_common_features = []
        for feature in common_features:
            if 'encoded' not in feature and feature != 'Stage':  # 排除编码特征和阶段特征
                numeric_common_features.append(feature)
        
        if not numeric_common_features:
            print("没有找到共同的数值型特征")
            return
        
        # 为每个共同数值特征创建分布图
        for feature in numeric_common_features:
            plt.figure(figsize=(15, 5))
            
            # 中风数据
            if feature in self.stroke_processed['X'].columns:
                plt.subplot(1, 3, 1)
                plt.hist(self.stroke_processed['X'][feature], bins=30, alpha=0.7, label='中风')
                plt.title(f'中风患者{feature_chinese_mapping.get(feature, feature)}分布')
                plt.xlabel(feature_chinese_mapping.get(feature, feature))
                plt.ylabel('频数')
                plt.legend()
            
            # 心脏病数据
            if feature in self.heart_processed['X'].columns:
                plt.subplot(1, 3, 2)
                plt.hist(self.heart_processed['X'][feature], bins=30, alpha=0.7, label='心脏病', color='orange')
                plt.title(f'心脏病患者{feature_chinese_mapping.get(feature, feature)}分布')
                plt.xlabel(feature_chinese_mapping.get(feature, feature))
                plt.ylabel('频数')
                plt.legend()
            
            # 肝硬化数据
            if feature in self.cirrhosis_processed['X'].columns:
                plt.subplot(1, 3, 3)
                plt.hist(self.cirrhosis_processed['X'][feature], bins=30, alpha=0.7, label='肝硬化', color='green')
                plt.title(f'肝硬化患者{feature_chinese_mapping.get(feature, feature)}分布')
                plt.xlabel(feature_chinese_mapping.get(feature, feature))
                plt.ylabel('频数')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, f'{feature}_分布比较.png'))
            plt.show()

    def analyze_common_numeric_features(self):
        """分析共同数值特征的分布"""
        print("\n--- 共同数值特征统计分析 ---")
        
        # 找出三种疾病共有的数值特征
        numeric_features = ['Age', 'Cholesterol']
        
        for feature in numeric_features:
            # 检查该特征是否存在于各疾病数据中
            stroke_exists = feature in self.stroke_processed['X'].columns
            heart_exists = feature in self.heart_processed['X'].columns
            cirrhosis_exists = feature in self.cirrhosis_processed['X'].columns
            
            if not (stroke_exists or heart_exists or cirrhosis_exists):
                continue
            
            print(f"\n特征: {feature_chinese_mapping.get(feature, feature)}")
            
            # 计算各疾病的统计信息
            if stroke_exists:
                stroke_stats = self.stroke_processed['X'][feature].describe()
                print(f"中风患者: 均值={stroke_stats['mean']:.2f}, 标准差={stroke_stats['std']:.2f}, 最小值={stroke_stats['min']:.2f}, 最大值={stroke_stats['max']:.2f}")
            
            if heart_exists:
                heart_stats = self.heart_processed['X'][feature].describe()
                print(f"心脏病患者: 均值={heart_stats['mean']:.2f}, 标准差={heart_stats['std']:.2f}, 最小值={heart_stats['min']:.2f}, 最大值={heart_stats['max']:.2f}")
            
            if cirrhosis_exists:
                cirrhosis_stats = self.cirrhosis_processed['X'][feature].describe()
                print(f"肝硬化患者: 均值={cirrhosis_stats['mean']:.2f}, 标准差={cirrhosis_stats['std']:.2f}, 最小值={cirrhosis_stats['min']:.2f}, 最大值={cirrhosis_stats['max']:.2f}")
            
            # 可视化血清胆固醇统计信息
            if feature == 'Cholesterol':
                self.visualize_cholesterol_stats(heart_stats, cirrhosis_stats)

    def visualize_cholesterol_stats(self, heart_stats, cirrhosis_stats):
        """可视化血清胆固醇统计信息"""
        diseases = ['心脏病', '肝硬化']
        means = [heart_stats['mean'], cirrhosis_stats['mean']]
        stds = [heart_stats['std'], cirrhosis_stats['std']]
        mins = [heart_stats['min'], cirrhosis_stats['min']]
        maxs = [heart_stats['max'], cirrhosis_stats['max']]
        
        x = np.arange(len(diseases))
        width = 0.2
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, means, width, label='均值')
        rects2 = ax.bar(x, stds, width, label='标准差')
        rects3 = ax.bar(x + width, mins, width, label='最小值')
        rects4 = ax.bar(x + 2 * width, maxs, width, label='最大值')
        
        ax.set_ylabel('血清胆固醇值')
        ax.set_title('心脏病和肝硬化患者血清胆固醇统计信息')
        ax.set_xticks(x + width)
        ax.set_xticklabels(diseases)
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_folder, '血清胆固醇统计信息可视化.png'))
        plt.show()

    def run_analysis(self):
        if not self.load_and_preprocess_data():
            return
        
        self.multi_disease_analysis()
        print("\n=== 分析完成 ===")


# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = DiseaseAnalyzer()
    
    # 运行分析
    analyzer.run_analysis()