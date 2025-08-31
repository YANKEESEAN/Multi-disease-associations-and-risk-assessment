import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import re

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False


# ----------------------
# 1. 数据加载与预处理
# ----------------------
# 数据集路径
stroke_path = r"F:\stroke.csv"
heart_path = r"F:\heart.csv"
cirrhosis_path = r"F:\cirrhosis.csv"

# 加载原始数据
stroke_raw = pd.read_csv(stroke_path)
heart_raw = pd.read_csv(heart_path)
cirrhosis_raw = pd.read_csv(cirrhosis_path)


def preprocess_data(df, df_raw, disease_type):
    """数据预处理且输出详细日志"""
    print(f"\n===== {disease_type}数据集预处理 =====")
    original_rows, original_cols = df_raw.shape
    
    # 处理缺失值
    if disease_type == 'stroke':
        # 中风数据集：处理特殊标记缺失值
        df['体重指数'] = df['体重指数'].replace('未知体重指数', np.nan)
        df['吸烟状态'] = df['吸烟状态'].replace('未知吸烟状态', '未知')
        df['体重指数'] = pd.to_numeric(df['体重指数'], errors='coerce')
        target_col = '是否中风'
        
        # 缺失值详细统计
        missing_stats = df.isnull().sum()[df.isnull().sum() > 0]
        print("\n缺失值统计：")
        for col, count in missing_stats.items():
            print(f"- {col}：{count}条（{count / original_rows:.2%}）")
        
        # 缺失值处理策略
        df['体重指数'] = df['体重指数'].fillna(df['体重指数'].median())
        print(f"缺失值处理：体重指数采用中位数填充")
    
    elif disease_type == 'heart':
        # 心脏病数据集：聚焦心血管风险因素处理
        target_col = '是否有心脏病'
        
        # 修复列名特殊字符
        df.columns = [re.sub(r'[\\/:\*\?"<>\|]', '_', col) for col in df.columns]
        
        # 缺失值统计
        missing_stats = df.isnull().sum()[df.isnull().sum() > 0]
        if not missing_stats.empty:
            print("\n缺失值统计：")
            for col, count in missing_stats.items():
                print(f"- {col}：{count}条（{count / original_rows:.2%}）")
        
        # 缺失值处理
        df = df.dropna()
        print(f"缺失值处理：删除包含缺失值的记录，剩余{df.shape[0]}条")
    
    elif disease_type == 'cirrhosis':
        # 肝硬化数据集：基于肝病结局定义目标变量
        df['是否肝硬化'] = df['状态'].apply(lambda x: '是' if x in ['死亡', '肝脏被审查'] else '否')
        target_col = '是否肝硬化'
        
        # 修复列名特殊字符
        df.columns = [re.sub(r'[\\/:\*\?"<>\|]', '_', col) for col in df.columns]
        
        # 缺失值统计与处理
        missing_stats = df.isnull().sum()[df.isnull().sum() > 0]
        if not missing_stats.empty:
            print("\n缺失值统计：")
            for col, count in missing_stats.items():
                print(f"- {col}：{count}条（{count / original_rows:.2%}）")
        
        # 数值列中位数填充+分类列众数填充
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns.drop(['状态', target_col], errors='ignore')
        
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
        print(f"缺失值处理：数值列中位数填充（{len(numeric_cols)}列），分类列众数填充（{len(cat_cols)}列）")
    
    # 分类变量编码及说明
    cat_cols = df.select_dtypes(include=['object']).columns.drop(target_col, errors='ignore')
    encode_maps = {}  # 存储编码映射关系
    
    for col in cat_cols:
        df[col], codes = pd.factorize(df[col])
        encode_maps[col] = dict(zip(codes, range(len(codes))))
    
    print(f"\n分类变量编码：共处理{len(cat_cols)}个特征（编码映射已保存）")
    
    # 预处理后数据统计
    final_rows = df.shape[0]
    print(f"\n预处理完成：{original_rows} → {final_rows}条记录（{final_rows / original_rows:.2%}保留率）")
    
    return df, target_col, encode_maps


# 执行预处理并保存编码映射
stroke_df, stroke_target, stroke_maps = preprocess_data(stroke_raw.copy(), stroke_raw, 'stroke')
heart_df, heart_target, heart_maps = preprocess_data(heart_raw.copy(), heart_raw, 'heart')
cirrhosis_df, cirrhosis_target, cirrhosis_maps = preprocess_data(cirrhosis_raw.copy(), cirrhosis_raw, 'cirrhosis')


# ----------------------
# 2. 描述性统计分析
# ----------------------
def descriptive_analysis(df, disease_name, target_col, encode_maps):
    """描述性统计分析，优化图表显示"""
    print(f"\n===== {disease_name}数据集描述性统计 =====")
    
    # 连续变量定义
    numeric_cols = df.select_dtypes(include=['float64']).columns.tolist()
    int_cols = df.select_dtypes(include=['int64']).columns.tolist()
    exclude_columns = ['唯一标识符', '患者 ID', '记录编号']
    
    for col in int_cols:
        if (col != target_col
                and col not in exclude_columns
                and df[col].nunique() > 10
                and col not in ['性别', '是否高血压', '是否高血糖', '吸烟状态']):
            numeric_cols.append(col)
    
    numeric_cols = list(set(numeric_cols))  # 去重
    
    # 明确分类变量
    categorical_cols = [col for col in df.columns
                        if col != target_col and
                        (df[col].dtype == 'int64' and df[col].nunique() <= 10
                         or df[col].dtype == 'object')]
    
    # 1. 核心指标概览
    print(f"\n【核心指标】")
    print(f"- 样本量：{df.shape[0]}条")
    disease_ratio = df[target_col].value_counts(normalize=True).get('是', 0)
    print(f"- 患病比例：{disease_ratio:.2%}")
    
    # 2. 连续变量分布可视化
    print("\n【连续变量统计描述】")
    if numeric_cols:  # 确保有连续变量再绘图
        print(df[numeric_cols].describe().round(2))
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols[:6]):  # 最多显示6个连续变量
            plt.subplot(2, 3, i + 1)
            # 绘制连续变量分布图
            ax = sns.histplot(
                data=df,
                x=col,
                hue=target_col,  # 按目标列分组
                hue_order=['是', '否'],  # 固定顺序
                kde=True,
                element='step',
                palette=['#e74c3c', '#3498db'],  # 固定颜色
                stat='density'
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                ['是', '否'],
                title=target_col,
                title_fontsize=10,
                fontsize=9,
                loc='upper right'
            )
            plt.title(f'{col}分布（按{target_col}分组）')
            plt.xlabel(col)
            plt.ylabel('密度')
        
        plt.tight_layout()
        safe_disease_name = re.sub(r'[\\/:\*\?"<>\|]', '_', disease_name)
        plt.savefig(f'{safe_disease_name}_连续变量分布.png')
        plt.close()
        print(f"连续变量分布图已保存：{safe_disease_name}_连续变量分布.png")
    else:
        print("未检测到有效连续变量，跳过连续变量分布图绘制")
    
    # 3. 离散变量分析
    print("\n【离散变量分布】")
    for col in categorical_cols[:3]:  # 最多显示3个分类变量
        print(f"\n{col}分布（编码映射：{encode_maps.get(col, '无')}）：")
        freq = df[col].value_counts(normalize=True).sort_index() * 100
        print(freq.round(1).astype(str) + '%')
        
        # 离散变量患病比例可视化
        plt.figure(figsize=(8, 5))
        # 计算不同分组的患病比例
        prop_df = df.groupby(col)[target_col].value_counts(normalize=True).unstack() * 100
        
        if col in encode_maps:
            reverse_map = {v: k for k, v in encode_maps[col].items()}
            prop_df.index = [reverse_map.get(idx, idx) for idx in prop_df.index]
            
            if col == '性别' and set(reverse_map.values()) == {'男', '女'}:
                prop_df = prop_df.reindex(['男', '女'])
        
        ax = prop_df[['是', '否']].plot(
            kind='bar',
            color=['#e74c3c', '#3498db'],
            width=0.6
        )
        ax.legend(title=target_col, labels=['是', '否'])
        plt.title(f'{col}分组的{target_col}比例')
        plt.xlabel(col)  # X轴显示变量名
        plt.ylabel('比例 (%)')
        plt.xticks(rotation=0)  # X轴标签水平显示
        plt.tight_layout()
        safe_col = re.sub(r'[\\/:\*\?"<>\|]', '_', col)
        plt.savefig(f'{safe_disease_name}_{safe_col}_患病比例.png')
        plt.close()
        print(f"{col}患病比例图已保存：{safe_disease_name}_{safe_col}_患病比例.png")


# 执行描述性分析
descriptive_analysis(stroke_df, '中风', stroke_target, stroke_maps)
descriptive_analysis(heart_df, '心脏病', heart_target, heart_maps)
descriptive_analysis(cirrhosis_df, '肝硬化', cirrhosis_target, cirrhosis_maps)


# ----------------------
# 3. 差异性检验
# ----------------------
def difference_test(df, disease_name, target_col):
    """差异性检验，增加统计显著性解读"""
    print(f"\n===== {disease_name}差异性检验 =====")
    df_temp = df.copy()
    df_temp[target_col] = df_temp[target_col].map({'是': 1, '否': 0})
    sick_group = df_temp[df_temp[target_col] == 1]
    healthy_group = df_temp[df_temp[target_col] == 0]
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = [col for col in df.select_dtypes(include=['int64']).columns
                       if col != target_col and df[col].nunique() < 10]
    
    # 连续变量检验结果整理
    significant_numeric = []
    for col in numeric_cols:
        if col == target_col:
            continue
        
        stat, p_norm = stats.shapiro(df[col])
        if p_norm > 0.05:
            stat, p_val = stats.ttest_ind(sick_group[col], healthy_group[col], equal_var=False)
            test_type = "T检验"
        else:
            stat, p_val = stats.mannwhitneyu(sick_group[col], healthy_group[col])
            test_type = "Mann-Whitney U检验"
        
        if p_val < 0.05:
            significant_numeric.append({
                '特征': col,
                '检验类型': test_type,
                'p值': p_val,
                '患病组均值': sick_group[col].mean(),
                '健康组均值': healthy_group[col].mean(),
                '效应大小': abs(sick_group[col].mean() - healthy_group[col].mean()) / df[col].std()
            })
    
    # 离散变量检验结果整理
    significant_cat = []
    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            stat, p_val, dof, expected = stats.chi2_contingency(contingency)
            if p_val < 0.05:
                # 计算Cramer's V效应大小
                n = contingency.sum().sum()
                v = np.sqrt(stat / (n * (min(contingency.shape) - 1)))
                significant_cat.append({
                    '特征': col,
                    '检验类型': "卡方检验",
                    'p值': p_val,
                    '效应大小': v,
                    '患病组占比': df[df[target_col] == '是'][col].value_counts(normalize=True).max()
                })
    
    # 结果输出与解读
    print("\n【显著差异特征】（p<0.05）")
    if significant_numeric:
        print("\n连续变量：")
        for item in sorted(significant_numeric, key=lambda x: x['p值'])[:3]:
            print(f"- {item['特征']}（{item['检验类型']}）：p={item['p值']:.3f}，"
                  f"效应大小={item['效应大小']:.2f}，"
                  f"患病组均值{item['患病组均值']:.2f} vs 健康组{item['健康组均值']:.2f}")
    
    if significant_cat:
        print("\n离散变量：")
        for item in sorted(significant_cat, key=lambda x: x['p值'])[:3]:
            print(f"- {item['特征']}（{item['检验类型']}）：p={item['p值']:.3f}，"
                  f"效应大小={item['效应大小']:.2f}，"
                  f"最高患病亚组占比{item['患病组占比']:.2%}")
    
    return significant_numeric + significant_cat


# 执行差异性检验并保存结果
stroke_significant = difference_test(stroke_df, '中风', stroke_target)
heart_significant = difference_test(heart_df, '心脏病', heart_target)
cirrhosis_significant = difference_test(cirrhosis_df, '肝硬化', cirrhosis_target)


# ----------------------
# 4. 相关性分析
# ----------------------
def cross_dataset_analysis(stroke_df, heart_df, cirrhosis_df, stroke_maps, heart_maps, cirrhosis_maps):
    """跨数据集对比分析，修复高血压列不存在的错误"""
    print(f"\n===== 跨数据集对比分析 =====")
    
    # 各数据集目标列
    stroke_target = '是否中风'
    heart_target = '是否有心脏病'
    cirrhosis_target = '是否肝硬化'
    
    # 1. 高血压患病率对比
    # 中风数据集
    stroke_hypertension = 0.0
    if '是否高血压' in stroke_df.columns:
        stroke_hypertension = stroke_df[stroke_df['是否高血压'] == 1][stroke_target].value_counts(
            normalize=True).get('是', 0) * 100
    
    # 心脏病数据集
    heart_hypertension = 0.0
    if '是否高血压' in heart_df.columns:
        heart_hypertension = heart_df[heart_df['是否高血压'] == 1][heart_target].value_counts(
            normalize=True).get('是', 0) * 100
    else:
        print("警告：心脏病数据集中未找到'是否高血压'列，相关统计值记为0")
    
    # 肝硬化数据集
    cirrhosis_hypertension = 0.0
    print("警告：肝硬化数据集中未找到'是否高血压'列，相关统计值记为0")
    
    # 2. 平均年龄对比
    stroke_age = stroke_df[stroke_df[stroke_target] == '是']['年龄'].mean() if '年龄' in stroke_df.columns else 0
    heart_age = heart_df[heart_df[heart_target] == '是']['年龄'].mean() if '年龄' in heart_df.columns else 0
    cirrhosis_age_col = '年龄(年)' if '年龄(年)' in cirrhosis_df.columns else '年龄'
    cirrhosis_age = cirrhosis_df[cirrhosis_df[cirrhosis_target] == '是'][
        cirrhosis_age_col].mean() if cirrhosis_age_col in cirrhosis_df.columns else 0
    
    # 3. 高血糖患病率对比
    stroke_diabetes = stroke_df[stroke_df['是否高血糖'] == 1][stroke_target].value_counts(
        normalize=True).get('是', 0) * 100 if '是否高血糖' in stroke_df.columns else 0
    
    heart_diabetes = 0.0
    if '是否高血糖' in heart_df.columns:
        heart_diabetes = heart_df[heart_df['是否高血糖'] == 1][heart_target].value_counts(
            normalize=True).get('是', 0) * 100
    else:
        print("警告：心脏病数据集中未找到'是否高血糖'列，相关统计值记为0")
    
    # 输出共同风险因素对比
    print("\n【共同风险因素对比】")
    print(f"高血压患病率：中风{stroke_hypertension:.1f}%，心脏病{heart_hypertension:.1f}%，肝硬化{cirrhosis_hypertension:.1f}%")
    print(f"患者平均年龄：中风{stroke_age:.1f}岁，心脏病{heart_age:.1f}岁，肝硬化{cirrhosis_age:.1f}岁")
    print(f"高血糖患病率：中风{stroke_diabetes:.1f}%，心脏病{heart_diabetes:.1f}%")
    
    # 4. 年龄分布对比
    plt.figure(figsize=(12, 6))
    if '年龄' in stroke_df.columns:
        sns.kdeplot(
            stroke_df[stroke_df[stroke_target] == '是']['年龄'],
            label=f'{stroke_target}-是',
            color='#e74c3c'
        )
        sns.kdeplot(
            stroke_df[stroke_df[stroke_target] == '否']['年龄'],
            label=f'{stroke_target}-否',
            color='#3498db'
        )
    
    if '年龄' in heart_df.columns:
        sns.kdeplot(
            heart_df[heart_df[heart_target] == '是']['年龄'],
            label=f'{heart_target}-是',
            color='#f39c12'
        )
        sns.kdeplot(
            heart_df[heart_df[heart_target] == '否']['年龄'],
            label=f'{heart_target}-否',
            color='#2ecc71'
        )
    
    if cirrhosis_age_col in cirrhosis_df.columns:
        sns.kdeplot(
            cirrhosis_df[cirrhosis_df[cirrhosis_target] == '是'][cirrhosis_age_col],
            label=f'{cirrhosis_target}-是',
            color='#9b59b6'
        )
        sns.kdeplot(
            cirrhosis_df[cirrhosis_df[cirrhosis_target] == '否'][cirrhosis_age_col],
            label=f'{cirrhosis_target}-否',
            color='#34495e'
        )
    
    plt.title('三种疾病患者与健康人群的年龄分布对比')
    plt.xlabel('年龄')
    plt.ylabel('密度')
    plt.legend(title='疾病状态', fontsize=9)
    plt.tight_layout()
    plt.savefig('三种疾病患者年龄分布对比.png')
    plt.close()
    print("三种疾病患者年龄分布对比图已保存")
    
    # 5. 性别差异分析
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    x = np.arange(2)  # 男/女两个类别位置
    
    # 中风数据集性别分布
    if '性别' in stroke_df.columns:
        stroke_gender = stroke_df.groupby('性别')[stroke_target].value_counts(normalize=True).unstack() * 100
        stroke_gender_map = {v: k for k, v in stroke_maps.get('性别', {}).items()}
        stroke_gender.index = [stroke_gender_map.get(idx, idx) for idx in stroke_gender.index]
        stroke_gender = stroke_gender.reindex(['男', '女'])
        plt.bar(x - bar_width, stroke_gender['是'], width=bar_width, label=stroke_target, color='#e74c3c')
    
    # 心脏病数据集性别分布
    if '性别' in heart_df.columns:
        heart_gender = heart_df.groupby('性别')[heart_target].value_counts(normalize=True).unstack() * 100
        heart_gender_map = {v: k for k, v in heart_maps.get('性别', {}).items()}
        heart_gender.index = [heart_gender_map.get(idx, idx) for idx in heart_gender.index]
        heart_gender = heart_gender.reindex(['男', '女'])
        plt.bar(x, heart_gender['是'], width=bar_width, label=heart_target, color='#f39c12')
    
    # 肝硬化数据集性别分布
    if '性别' in cirrhosis_df.columns:
        cirrhosis_gender = cirrhosis_df.groupby('性别')[cirrhosis_target].value_counts(normalize=True).unstack() * 100
        cirrhosis_gender_map = {v: k for k, v in cirrhosis_maps.get('性别', {}).items()}
        cirrhosis_gender.index = [cirrhosis_gender_map.get(idx, idx) for idx in cirrhosis_gender.index]
        cirrhosis_gender = cirrhosis_gender.reindex(['男', '女'])
        plt.bar(x + bar_width, cirrhosis_gender['是'], width=bar_width, label=cirrhosis_target, color='#9b59b6')
    
    plt.title('不同性别中的疾病患病率对比')
    plt.xlabel('性别')
    plt.ylabel('患病率 (%)')
    plt.xticks(x, ['男', '女'])
    plt.legend(title='疾病类型', fontsize=9)
    plt.tight_layout()
    plt.savefig('三种疾病患者性别分布对比.png')
    plt.close()
    print("三种疾病患者性别分布对比图已保存")
    
    # 6. 多疾病共同影响因素
    stroke_factors = [item['特征'] for item in stroke_significant]
    heart_factors = [item['特征'] for item in heart_significant]
    cirrhosis_factors = [item['特征'] for item in cirrhosis_significant]
    
    common_factors = []
    for factor in set(stroke_factors + heart_factors + cirrhosis_factors):
        count = 0
        diseases = []
        if factor in stroke_factors:
            count += 1
            diseases.append('中风')
        if factor in heart_factors:
            count += 1
            diseases.append('心脏病')
        if factor in cirrhosis_factors:
            count += 1
            diseases.append('肝硬化')
        
        if count >= 2:
            common_factors.append({'特征': factor, '影响疾病': diseases})
    
    print("\n【多疾病共同影响因素】")
    if common_factors:
        for f in common_factors:
            print(f"- {f['特征']}：影响{f['影响疾病']}")
    else:
        print("未发现两种及以上疾病共有的显著影响因素")


# 执行跨数据集对比分析
cross_dataset_analysis(stroke_df, heart_df, cirrhosis_df, stroke_maps, heart_maps, cirrhosis_maps)

print("\n===== 数据分析完成 =====")
print("所有图表已保存为PNG文件")
