
"""
最终版：数据预处理主程序
"""
import os
import numpy as np
import pandas as pd
from scipy import signal

# ====================== 配置路径 ======================
normal_data_path = r"E:\BaiduNetdiskDownload\database\Train_set\normal_data"
normal_tag_path = r"E:\BaiduNetdiskDownload\database\Train_set\normal_tag"
Hypnogram_data_path = r"E:\BaiduNetdiskDownload\database\Train_set\Hypnogram_data"
Hypnogram_tag_path = r"E:\BaiduNetdiskDownload\database\Train_set\Hypnogram_tag"

# ====================== 读取文件并过滤掉非原始数据文件 ======================
normal_eeg_files = os.listdir(normal_data_path)
normal_label_files = os.listdir(normal_tag_path)
hypnogram_eeg_files = os.listdir(Hypnogram_data_path)
hypnogram_label_files = os.listdir(Hypnogram_tag_path)

# 合并所有文件
all_files = normal_eeg_files + normal_label_files + hypnogram_eeg_files + hypnogram_label_files

# 分类文件，但要过滤掉已经处理过的文件
eeg_files = []
label_files = []

for file_name in all_files:
    if "EEG" in file_name and "Cz" in file_name and "filtered" not in file_name.lower():
        eeg_files.append(file_name)
    elif "Hypnogram" in file_name and "Data" in file_name:
        label_files.append(file_name)

print(f"脑电文件数量: {len(eeg_files)} (已过滤处理过的文件)")
print(f"标签文件数量: {len(label_files)}")


# ====================== 定义匹配函数 ======================
def get_match_key(file_name):
    """改进的匹配键生成函数"""
    # 移除扩展名
    name_clean = file_name.replace('.txt', '')

    # 按下划线分割
    parts = name_clean.split('_')

    # 寻找被试ID（通常是字母数字组合，长度在6-10之间）
    subject_id = None
    part_info = None

    for part in parts:
        if len(part) >= 6 and part.isalnum():  # 至少6个字符且全是字母数字
            subject_id = part
            break

    # 寻找Part信息
    for i, part in enumerate(parts):
        if part == "Part" and i + 1 < len(parts):
            part_num = parts[i + 1]
            part_info = f"Part_{part_num}"
            break

    if subject_id and part_info:
        return f"{subject_id}_{part_info}"
    else:
        return name_clean


# ====================== 执行匹配 ======================
# 构建标签文件的匹配字典
label_key_dict = {}
for label_file in label_files:
    key = get_match_key(label_file)
    label_key_dict[key] = label_file

# 遍历脑电文件，匹配对应的标签文件
matched_data = []
unmatched_files = []

for eeg_file in eeg_files:
    key = get_match_key(eeg_file)
    if key in label_key_dict:
        # 匹配成功
        label_file = label_key_dict[key]
        matched_data.append({
            "eeg_file": eeg_file,
            "label_file": label_file,
            "match_key": key
        })
    else:
        unmatched_files.append(eeg_file)

print(f"成功匹配: {len(matched_data)} 组")
if unmatched_files:
    print(f"未匹配: {len(unmatched_files)} 个文件")
    for f in unmatched_files:
        print(f"  {f}")

# ====================== 读取标签文件，筛选有效标签 ======================
valid_labels = ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage R", "W", "1", "2", "3",
                "R"]

# 用于存储所有文件的处理结果
all_processed_data = []

if matched_data:
    for idx, match_item in enumerate(matched_data, 1):
        eeg_file = match_item["eeg_file"]
        label_file = match_item["label_file"]
        match_key = match_item["match_key"]

        # 根据文件名判断属于哪个文件夹
        if eeg_file in os.listdir(normal_data_path):
            eeg_file_path = os.path.join(normal_data_path, eeg_file)
        else:
            eeg_file_path = os.path.join(Hypnogram_data_path, eeg_file)

        if label_file in os.listdir(normal_tag_path):
            label_file_path = os.path.join(normal_tag_path, label_file)
        else:
            label_file_path = os.path.join(Hypnogram_tag_path, label_file)

        # 读取脑电数据
        try:
            eeg_data = np.loadtxt(eeg_file_path)
        except Exception as e:
            print(f"读取脑电数据失败 {eeg_file_path}: {e}")
            continue

        # 读取标签文件
        try:
            label_data = pd.read_csv(label_file_path, sep='\t', header=0)

            # 重命名列以匹配预期格式
            column_mapping = {
                'onset_sec': 'start_time',
                'end_sec': 'end_time',
                'duration_sec': 'duration',
                'description': 'label'
            }

            actual_columns = {}
            for old_col, new_col in column_mapping.items():
                if old_col in label_data.columns:
                    actual_columns[old_col] = new_col

            if actual_columns:
                label_data.rename(columns=actual_columns, inplace=True)
            else:
                print(f"错误：标签文件 {label_file_path} 缺少必要的列")
                continue

        except Exception as e:
            print(f"读取标签文件失败 {label_file_path}: {e}")
            continue

        # 筛选有效标签
        clean_label_data = label_data[label_data["label"].isin(valid_labels)]

        # 标准化标签
        label_mapping = {
            "Sleep stage W": "W",
            "Sleep stage 1": "1",
            "Sleep stage 2": "2",
            "Sleep stage 3": "3",
            "Sleep stage R": "R"
        }
        clean_label_data['label'] = clean_label_data['label'].map(lambda x: label_mapping.get(x, x))

        # 验证时长匹配
        eeg_duration = len(eeg_data) / 100  # 采样频率100Hz
        label_total_duration = clean_label_data["duration"].sum()

        # 存储当前文件的处理结果
        all_processed_data.append({
            "match_key": match_key,
            "eeg_file": eeg_file,
            "eeg_data": eeg_data,
            "label_file": label_file,
            "raw_label": label_data,
            "clean_label": clean_label_data,
            "eeg_duration": eeg_duration,
            "label_total_duration": label_total_duration
        })

    print(f"批量处理完成！共处理 {len(all_processed_data)} 组有效数据-标签文件")
else:
    print("无匹配成功的文件，无法进行标签清洗！")

print("\n第二步任务完成：批量读取+数据-标签匹配+标签清洗！")


# ====================== 脑电信号滤波 + 30秒帧分割 ======================
def filter_eeg_signal(eeg_data, fs=100):
    """
    对脑电信号进行滤波：0.5-30Hz带通滤波 + 50Hz陷波滤波
    """
    # 50Hz陷波滤波
    f0 = 50.0
    Q = 30.0
    b, a = signal.iirnotch(f0, Q, fs)
    eeg_notch = signal.filtfilt(b, a, eeg_data)

    # 0.5-30Hz带通滤波
    low = 0.5
    high = 30.0
    b, a = signal.butter(4, [low, high], btype='bandpass', fs=fs)
    eeg_filtered = signal.filtfilt(b, a, eeg_notch)

    return eeg_filtered


def split_eeg_into_frames(eeg_data, clean_label, fs=100, frame_duration=30):
    """
    将滤波后的脑电数据按30秒/帧分割
    """
    frame_points = fs * frame_duration  # 每帧3000个点
    frames = []
    labels = []

    for idx, row in clean_label.iterrows():
        start_sec = float(row["start_time"])
        end_sec = float(row["end_time"])

        start_idx = int(np.floor(start_sec * fs))
        end_idx = int(np.floor(end_sec * fs))

        if start_idx < 0 or end_idx > len(eeg_data):
            continue

        frame_length = end_idx - start_idx
        if frame_length == frame_points:
            frame = eeg_data[start_idx:end_idx]
            frames.append(frame)
            labels.append(row["label"])
        elif frame_length > frame_points:
            # 帧太长，取前3000个点
            frame = eeg_data[start_idx:start_idx + frame_points]
            frames.append(frame)
            labels.append(row["label"])
        elif frame_length < frame_points and frame_length > 0:
            # 帧太短，用零填充
            frame = np.zeros(frame_points)
            frame[:frame_length] = eeg_data[start_idx:end_idx]
            frames.append(frame)
            labels.append(row["label"])

    return np.array(frames), np.array(labels)


# 批量处理：滤波 + 帧分割
final_train_data = []

if all_processed_data:
    for idx, item in enumerate(all_processed_data, 1):
        match_key = item["match_key"]
        raw_eeg = item["eeg_data"]
        clean_label = item["clean_label"]

        # 滤波
        filtered_eeg = filter_eeg_signal(raw_eeg)

        # 30秒帧分割
        frames, frame_labels = split_eeg_into_frames(filtered_eeg, clean_label)

        # 存储最终结果
        final_train_data.append({
            "match_key": match_key,
            "filtered_eeg": filtered_eeg,
            "eeg_frames": frames,
            "frame_labels": frame_labels,
            "frame_count": len(frames)
        })

    # 打印总统计
    total_frames = sum([item["frame_count"] for item in final_train_data])
    print(f"\n📊 滤波+帧分割全量处理完成！")
    print(f"总处理文件组数：{len(final_train_data)}")
    print(f"总有效30秒帧数量：{total_frames}")
    print(f"每帧数据点：{30 * 100}个（符合要求）")
else:
    print("❌ 无预处理数据，无法进行滤波和帧分割！")

print("\n第三步任务完成：脑电滤波 + 30秒帧分割！")

# ====================== 最终统计 ======================
print("\n" + "=" * 60)
print("数据预处理最终统计:")
print(f"- 总匹配文件对数: {len(matched_data)}")
print(f"- 成功处理的文件对数: {len(all_processed_data)}")
print(f"- 生成的30秒帧总数: {sum([item['frame_count'] for item in final_train_data])}")
print(
    f"- 平均每个文件产生帧数: {np.mean([item['frame_count'] for item in final_train_data]):.2f}" if final_train_data else 0)
print("=" * 60)

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# ====================== 特征提取 ======================
def extract_features(frame):
    """从单个30秒EEG帧提取特征"""
    features = {}

    # 时域特征
    features['mean'] = np.mean(frame)
    features['std'] = np.std(frame)
    features['var'] = np.var(frame)
    features['max'] = np.max(frame)
    features['min'] = np.min(frame)
    features['range'] = features['max'] - features['min']
    features['skewness'] = np.mean(((frame - features['mean']) / features['std']) ** 3) if features['std'] != 0 else 0
    features['kurtosis'] = np.mean(((frame - features['mean']) / features['std']) ** 4) if features['std'] != 0 else 0

    # 频域特征
    freqs, psd = welch(frame, fs=100, nperseg=256)

    # 在不同频段的能量
    delta = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])  # Delta (0.5-4 Hz)
    theta = np.sum(psd[(freqs >= 4) & (freqs <= 8)])  # Theta (4-8 Hz)
    alpha = np.sum(psd[(freqs >= 8) & (freqs <= 13)])  # Alpha (8-13 Hz)
    beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])  # Beta (13-30 Hz)

    features['delta_power'] = delta
    features['theta_power'] = theta
    features['alpha_power'] = alpha
    features['beta_power'] = beta

    # 相对功率
    total_power = delta + theta + alpha + beta
    if total_power > 0:
        features['delta_rel'] = delta / total_power
        features['theta_rel'] = theta / total_power
        features['alpha_rel'] = alpha / total_power
        features['beta_rel'] = beta / total_power
    else:
        features['delta_rel'] = features['theta_rel'] = features['alpha_rel'] = features['beta_rel'] = 0

    return features


# 提取所有帧的特征
print("正在提取特征...")
all_features = []
all_labels = []

for item in final_train_data:
    for i, frame in enumerate(item['eeg_frames']):
        features = extract_features(frame)
        all_features.append(features)
        all_labels.append(item['frame_labels'][i])

print(f"提取了 {len(all_features)} 个样本的特征")

# ====================== 数据准备 ======================
feature_matrix = np.array([[v for v in feat.values()] for feat in all_features])
labels_array = np.array(all_labels)

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(labels_array)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_matrix)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"标签类别: {le.classes_}")

# ====================== 多种模型训练 ======================
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}

print("\n开始训练多个模型...")
for name, model in models.items():
    print(f"训练 {name}...")

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }

    print(f"{name} 准确率: {accuracy:.4f}")

# ====================== 结果比较 ======================
print("\n" + "=" * 60)
print("模型性能比较:")
print("=" * 60)
for name, result in results.items():
    print(f"{name:20s}: {result['accuracy']:.4f}")

# 找出最佳模型
best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']

print(f"\n最佳模型: {best_model_name} (准确率: {results[best_model_name]['accuracy']:.4f})")

# ====================== 详细评估最佳模型 ======================
best_predictions = results[best_model_name]['predictions']

print(f"\n{best_model_name} 详细分类报告:")
print(classification_report(y_test, best_predictions, target_names=le.classes_))

# ====================== 混淆矩阵可视化 ======================
plt.figure(figsize=(15, 10))

# 模型性能比较柱状图
plt.subplot(2, 3, 1)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
bars = plt.bar(model_names, accuracies)
plt.title('模型准确率比较')
plt.ylabel('准确率')
plt.xticks(rotation=45)
# 在柱子上显示数值
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f'{acc:.3f}', ha='center', va='bottom')

# 最佳模型混淆矩阵
plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'{best_model_name} 混淆矩阵')

# 特征重要性（如果是随机森林）
if best_model_name == 'Random Forest':
    plt.subplot(2, 3, 3)
    feature_names = list(all_features[0].keys())
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # 取前10个重要特征

    plt.bar(range(len(indices)), importances[indices])
    plt.title('Top 10 特征重要性')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)

# 预测概率分布
plt.subplot(2, 3, 4)
unique, counts = np.unique(best_predictions, return_counts=True)
plt.bar([le.classes_[i] for i in unique], counts)
plt.title('预测结果分布')
plt.xlabel('睡眠阶段')
plt.ylabel('数量')

# 实际标签分布
plt.subplot(2, 3, 5)
unique_true, counts_true = np.unique(y_test, return_counts=True)
plt.bar([le.classes_[i] for i in unique_true], counts_true)
plt.title('真实标签分布')
plt.xlabel('睡眠阶段')
plt.ylabel('数量')

# 预测vs真实对比
plt.subplot(2, 3, 6)
plt.scatter(y_test, best_predictions, alpha=0.5)
plt.plot([0, len(le.classes_) - 1], [0, len(le.classes_) - 1], 'r--', lw=2)
plt.title('预测值 vs 真实值')
plt.xlabel('真实值')
plt.ylabel('预测值')

plt.tight_layout()
plt.show()

# ====================== 保存模型和预处理器 ======================
import joblib

# 保存最佳模型和预处理器
joblib.dump(best_model, 'best_sleep_classifier.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print(f"\n模型已保存为 'best_sleep_classifier.pkl'")
print(f"特征缩放器已保存为 'feature_scaler.pkl'")
print(f"标签编码器已保存为 'label_encoder.pkl'")

# ====================== 模型解释 ======================
print(f"\n{'=' * 60}")
print("模型训练完成总结:")
print(f"{'=' * 60}")
print(f"• 训练样本数: {len(X_train)}")
print(f"• 测试样本数: {len(X_test)}")
print(f"• 特征数量: {X_train.shape[1]}")
print(f"• 睡眠阶段类别: {list(le.classes_)}")
print(f"• 最佳模型: {best_model_name}")
print(f"• 测试集准确率: {results[best_model_name]['accuracy']:.4f}")
print(f"• 模型已保存，可用于后续预测")
