import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
        return SVC(C=2, kernel='rbf', class_weight='balanced', probability=True)
    elif model_type == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights='uniform')
    elif model_type == "dt":
        return DecisionTreeClassifier(criterion='gini', class_weight='balanced', random_state=1)
    elif model_type == "lr":
        return LogisticRegression(C=1, solver='lbfgs', max_iter=1000, class_weight='balanced')
    elif model_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=1000, random_state=1)

def evaluate_ml_pipeline(csv_path, model_type="rf"):
    print(f"\n{'='*60}")
    print(f"📊 ĐANG ĐÁNH GIÁ MÔ HÌNH ML: {model_type.upper()}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    df = clean_feature_matrix(df)
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
    
    print(f"Tổng số mẫu: {len(df)} | Số đặc trưng: {len(feature_cols)}")
    print(f"Phân phối nhãn: {dict(pd.Series(y).value_counts().rename(index={0:'No Stress', 1:'Stress'}))}\n")

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
        
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        accumulated_cm += cm
        
        metrics_list.append({
            'Fold': i + 1,
            'Test_Drive': test_drive,
            'Accuracy': acc,
            'Recall (Sens)': rec,
            'F1_Score': f1,
            'Time': end_time - start_time
        })
        
        print(f"Fold {i+1:02d} | Test: {test_drive:10} | Acc: {acc:.4f} | Sens: {rec:.4f} | F1: {f1:.4f}")

    results_df = pd.DataFrame(metrics_list)
    print("\n" + "-"*80)
    print("📈 KẾT QUẢ TRUNG BÌNH (CROSS-VALIDATION)")
    print("-" * 80)
    print(results_df[['Accuracy', 'Recall (Sens)', 'F1_Score']].mean().to_string())
    print("-" * 80)
    
    plot_confusion_matrix(accumulated_cm)
    
    return results_df

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 7))
    labels = ['Relax (0)', 'Stress (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels)
    plt.title('Accumulated Confusion Matrix (3-Class LODO)', fontweight='bold', pad=20)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ml_results = evaluate_ml_pipeline("./data/extracted_features/ml_features_dataset.csv", model_type="rf")