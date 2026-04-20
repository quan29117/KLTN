import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import time

def clean_feature_matrix(df):
    """
    Imputes missing NaN values with the column's median.
    Ensures compatibility with scikit-learn ML models.
    """
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"    [Impute] Filled NaNs in '{col}' with median: {median_val:.4f}")
            
    return df_clean

def get_model(model_type):
    if model_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,
            leaf_size=30,
            metric='minkowski',
            weights='uniform'
        )
    
    elif model_type == "svm":
        return SVC(
            C=2,
            kernel='rbf',
            cache_size=200,
            tol=0.001,
            class_weight='balanced'
        )
    
    elif model_type == "dt":
        return DecisionTreeClassifier(
            criterion='gini',
            min_samples_leaf=1,
            min_samples_split=2,
            splitter='best',
            class_weight='balanced'
        )
    
    elif model_type == "lr":
        return LogisticRegression(
            C=1,
            solver='lbfgs',
            tol=0.0001,
            class_weight='balanced',
            max_iter=1000
        )
    
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=110,
            criterion='entropy',
            random_state=1,
            class_weight='balanced',
            n_jobs=-1
        )
    
    elif model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(50,),
            activation='relu',
            momentum=0.9,
            solver='adam',
            max_iter=1000,
            random_state=1
        )
    
    else:
        raise ValueError(f"Model type {model_type} not supported.")

def evaluate_baseline_model(model_type="rf"):
    print(f"\n{'='*60}")
    print("📊 EVALUATING ML: BINARY CLASSIFICATION")
    print(f"{'='*60}")
    
    df = pd.read_csv("./data/extracted_features/final_feature_matrix.csv")
    df_clean = clean_feature_matrix(df)
    if df_clean['Stress_Binary'].dtype == object:
        df_clean['label_num'] = df_clean['Stress_Binary'].map({'no': 0, 'stressed': 1})
    else:
        df_clean['label_num'] = df_clean['Stress_Binary']

    ignore_cols = ['Stress_Binary', 'label_num', 'Drive', 'Window_Start', 'time', 'is_valid']
    feature_cols = [col for col in df_clean.columns if col not in ignore_cols]

    X = df_clean[feature_cols].values
    y = df_clean['label_num'].values
    groups = df_clean['Drive'].values

    unique_labels, counts = np.unique(y, return_counts=True)
    print("-" * 50)
    print("📋 LABEL DISTRIBUTION")
    print("-" * 50)
    for label, count in zip(unique_labels, counts):
        name = 'Stress (1)' if label == 1 else 'No Stress (0)'
        print(f"  [+] {name}: {count} samples ({(count/len(y)):.2%})")
    print(f"  [+] Total samples: {len(y)}")
    print(f"  [+] Total features: {len(feature_cols)}")
    print("-" * 50)

    cv = LeaveOneGroupOut()
    metrics_list = []
    accumulated_cm = np.zeros((2, 2), dtype=int)

    model = get_model(model_type)

    print("\nStarting Leave-One-Group-Out Cross-Validation...\n")

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        test_drive = np.unique(groups[test_idx])[0]

        # Standardize features (Fit on train, transform on both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and Predict
        start_train = time.time()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        end_train = time.time()
        train_time = end_train - start_train

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        accumulated_cm += cm
        tp, fn, fp, tn = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics_list.append({
            'Fold': i + 1,
            'Test_Drive': test_drive,
            'Accuracy': acc,
            'Precision': prec,
            'Recall_Sens': rec,
            'Specificity': spec,
            'F1_Score': f1,
            'Train_Time_Sec': train_time,
        })

        print(f"  Fold {i+1:02d} | Test: {test_drive} | Acc: {acc:.4f} | Sens: {rec:.4f} | Spec: {spec:.4f} | F1: {f1:.4f}")

    # Generate Final Report
    results = pd.DataFrame(metrics_list)
    avg_train = results['Train_Time_Sec'].mean()
    
    print("\n" + "-"*80)
    print("📈 AVERAGE PERFORMANCE METRICS (CROSS-VALIDATION)")
    print("-" * 80)
    print(results[['Accuracy', 'Precision', 'Recall_Sens', 'Specificity', 'F1_Score']].mean().to_string())
    print(f"Average Training Time:   {avg_train:.4f} seconds")
    print("-" * 80)

    # Plot Accumulated Confusion Matrix
    plt.figure(figsize=(6, 5))
    labels_display = ['Stress (1)', 'No Stress (0)']
    
    ax = sns.heatmap(
        accumulated_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels_display, yticklabels=labels_display,
        cbar_kws={'label': 'Number of Windows'}
    )
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.title('Accumulated Confusion Matrix (All Drives)', y=1.15, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    return results, model

ml_results, trained_model = evaluate_baseline_model(model_type="rf")