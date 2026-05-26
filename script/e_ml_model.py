import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def clean_feature_matrix(df):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    return df_clean

def apply_subject_normalization(df):
    """
    Chuẩn hóa đặc trưng theo từng Driver.
    Lấy giá trị hiện tại trừ đi trung bình của giai đoạn Rest1 của chính Driver đó.
    """
    normalized_dfs = []
    
    for drive_id, group in df.groupby('drive_id'):
        baseline_df = group[group['stage'].str.contains('Rest1', case=False, na=False)]
        if baseline_df.empty:
            baseline_df = group[group['label'] == 0]
            
        if not baseline_df.empty:
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            feat_cols = [c for c in numeric_cols if c not in ['label', 'target']]
            
            baseline_values = baseline_df[feat_cols].mean()
            group_norm = group.copy()
            group_norm[feat_cols] = group[feat_cols] - baseline_values
            normalized_dfs.append(group_norm)
        else:
            normalized_dfs.append(group)
            
    return pd.concat(normalized_dfs).reset_index(drop=True)

def get_ml_model(model_type):
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=110, criterion='entropy', random_state=1, class_weight='balanced', n_jobs=-1)
    elif model_type == "svm":
        return SVC(C=2, kernel='rbf', cache_size=200, tol=0.001, class_weight='balanced', probability=True)
    elif model_type == "knn":
        return KNeighborsClassifier(n_neighbors=5, leaf_size=30, metric='minkowski', weights='uniform')
    elif model_type == "dt":
        return DecisionTreeClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=2, splitter='best', class_weight='balanced', random_state=1)
    elif model_type == "lr":
        return LogisticRegression(C=1, solver='lbfgs', tol=0.0001, max_iter=1000, class_weight='balanced')
    elif model_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(50,), activation='relu', momentum=0.9, solver='adam', max_iter=1000, random_state=1)

def evaluate_ml_pipeline(csv_path, model_type="rf", use_norm=True):
    norm_status = "WITH_NORM" if use_norm else "WITHOUT_NORM"
    print(f"\n{'='*60}")
    print(f"E. First layer - YES / NO Stressed with {model_type.upper()} ({norm_status.replace('_', ' ')})")
    
    df = pd.read_csv(csv_path)
    df = clean_feature_matrix(df)
    
    if use_norm:
        df = apply_subject_normalization(df)
    
    df['target'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    ignore_cols = ['window_id', 'drive_id', 'label', 'stage', 'target']
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    groups = df['drive_id'].values 
    
    logo = LeaveOneGroupOut()
    metrics_list = []
    accumulated_cm = np.zeros((2, 2), dtype=int)
    
    print(f"Total: {len(df)} | No of extracted features: {len(feature_cols)}")
    print(f"Label distribution: {dict(pd.Series(y).value_counts().rename(index={0:'No Stress', 1:'Stress'}))}\n")

    for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_drive = groups[test_idx][0]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = get_ml_model(model_type)
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        end_time = time.time()
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        accumulated_cm += cm
        
        tn, fp = cm[1, 1], cm[1, 0]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_list.append({
            'Fold': i + 1,
            'Test_Drive': test_drive,
            'Accuracy': acc,
            'Recall (Sens)': rec,
            'Precision': prec,
            'Specificity': spec,
            'F1_Score': f1,
            'Time': end_time - start_time
        })
        
        print(f"Fold {i+1:02d} | Test: {test_drive:10} | Acc: {acc:.4f} | Sens: {rec:.4f} | Prec: {prec:.4f} | Spec: {spec:.4f} | F1: {f1:.4f}")

    results_df = pd.DataFrame(metrics_list)
    print("-" * 60)
    print(results_df[['Accuracy', 'Recall (Sens)', 'Precision', 'Specificity', 'F1_Score']].mean().to_string())
    print("-" * 60)
    
    plot_confusion_matrix(accumulated_cm, output=f"./result/cm_{model_type}_{norm_status.lower()}.png")
    
    return results_df

def plot_confusion_matrix(cm, output):
    plt.figure(figsize=(8, 8))
    labels = ['Stress (1)', 'Relax (0)']
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels, annot_kws={"size": 28, "weight": "bold"}, cbar=False)
    
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    
    ax.set_xticklabels(labels, fontsize=18, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=18, fontweight='bold')
    
    # plt.title('Accumulated Confusion Matrix', fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontweight='bold', fontsize=20, labelpad=20)
    plt.ylabel('True Label', fontweight='bold', fontsize=20, labelpad=20)
    plt.tight_layout()
    output_obj = Path(output)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=400, bbox_inches='tight')
    # plt.show()
    plt.close()
    
def plot_normalization_comparison(with_norm_df, without_norm_df, save_path="./result/norm_comparison.png"):
    """
    Vẽ biểu đồ cột nhóm so sánh chi tiết các thông số từ 2 kết quả DataFrame (with_norm và without_norm).
    """
    metrics_labels = ['Accuracy', 'Sensitivity\n(Recall)', 'Precision', 'Specificity', 'F1-Score']
    keys = ['Accuracy', 'Recall (Sens)', 'Precision', 'Specificity', 'F1_Score']
    
    mean_without = without_norm_df[keys].mean()
    mean_with = with_norm_df[keys].mean()
    
    values_without = [mean_without[k] for k in keys]
    values_with = [mean_with[k] for k in keys]
    
    x = np.arange(len(metrics_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, values_without, width, label='Without Normalization (Raw Features)', color='#e74c3c')
    rects2 = ax.bar(x + width/2, values_with, width, label='With Normalization (Subject Baseline)', color='#2ecc71')

    ax.set_ylabel('Scores', fontweight='bold', fontsize=12)
    ax.set_title('Comparison of Model Performance With and Without Subject Normalization', fontweight='bold', fontsize=14, pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontweight='bold', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height*100:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_obj, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def run():
    ml_results = evaluate_ml_pipeline("./data/extracted_features/ml_features_dataset.csv", model_type="rf", use_norm=True)

if __name__ == "__main__":
    FEATURES_CSV = "./data/extracted_features/ml_features_dataset.csv"
    
    metrics_without_norm = evaluate_ml_pipeline(FEATURES_CSV, model_type="rf", use_norm=False)
    metrics_with_norm = evaluate_ml_pipeline(FEATURES_CSV, model_type="rf", use_norm=True)