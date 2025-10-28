import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

dataset_path = input("Please enter the path to your dataset CSV file: ").strip()
target_col = input("What is the name of the target column? (Leave blank to use the last column): ").strip()

n_splits = int(input("How many cross-validation folds should we use? (k, default is 5): ") or 5)
n_estimators = int(input("How many base decision trees should each model have? (default is 50): ") or 50)
criterion = input("Which impurity measure do you want for the trees? (gini/entropy, default is gini): ").strip() or "gini"
max_depth_input = input("What's the max depth for the trees? (Leave blank for no limit): ").strip()
max_depth = int(max_depth_input) if max_depth_input != "" else None

print("\nLoading your dataset...")
data = pd.read_csv(dataset_path)

if not target_col:
    target_col = data.columns[-1]

if 'Id' in data.columns:
    print("Found and dropped an 'Id' column.")
    data = data.drop(columns=['Id'])

X = data.drop(columns=[target_col])
y = data[target_col]

if y.dtype == object or not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"The target column '{target_col}' has been label-encoded.")
    print(f"Classes found: {list(le.classes_)}")

non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"\nWarning: Found non-numeric feature columns: {non_numeric_cols}")
    print("Attempting to convert them to numbers. Columns that can't be converted will be dropped.")
    for col in non_numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.dropna(axis=1)

X = X.values
y = np.array(y)

print(f"\nDataset '{dataset_path}' is ready.")
print(f"Features being used: {list(data.drop(columns=[target_col]).columns)}")
print(f"Target variable: {target_col}")
print(f"Data shape: {X.shape}, with {len(np.unique(y))} classes.")

base_dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)

def make_bagging(estimator, n_estimators, random_state=42):
    try:
        return BaggingClassifier(estimator=estimator, n_estimators=n_estimators, random_state=random_state)
    except TypeError:
        return BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators, random_state=random_state)

def make_adaboost(estimator, n_estimators, random_state=42):
    try:
        return AdaBoostClassifier(estimator=estimator, n_estimators=n_estimators, random_state=random_state)
    except TypeError:
        return AdaBoostClassifier(base_estimator=estimator, n_estimators=n_estimators, random_state=random_state)

bagging_model = make_bagging(base_dt, n_estimators)
boosting_model = make_adaboost(base_dt, n_estimators)
stacking_model = StackingClassifier(estimators=[('bag', bagging_model), ('boost', boosting_model)], final_estimator=LogisticRegression(), n_jobs=-1)

def evaluate_model(model, X, y, k=5, model_name="Model"):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    final_cm = None
    
    print(f"\n{'='*15} Evaluating: {model_name} {'='*15}")

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\n--- Fold {i+1}/{k} ---")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_acc = accuracy_score(y_test, y_pred)
        fold_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        fold_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        fold_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"  Accuracy: {fold_acc:.4f}")
        print(f"  Precision: {fold_prec:.4f}")
        print(f"  Recall: {fold_rec:.4f}")
        print(f"  F1 Score: {fold_f1:.4f}")
        print("  Confusion Matrix:")
        print(cm)

        all_acc.append(fold_acc)
        all_prec.append(fold_prec)
        all_rec.append(fold_rec)
        all_f1.append(fold_f1)
        final_cm = cm

    print(f"\nFinal Confusion Matrix for {model_name}:")
    print(final_cm)

    return {
        "Accuracy": np.mean(all_acc),
        "Precision": np.mean(all_prec),
        "Recall": np.mean(all_rec),
        "F1 Score": np.mean(all_f1)
    }

print("\nStarting the evaluation process...")

bagging_results = evaluate_model(bagging_model, X, y, k=n_splits, model_name="Bagging")
boosting_results = evaluate_model(boosting_model, X, y, k=n_splits, model_name="Boosting")
stacking_results = evaluate_model(stacking_model, X, y, k=n_splits, model_name="Stacking")

summary_results = pd.DataFrame([bagging_results, boosting_results, stacking_results], index=["Bagging", "Boosting", "Stacking"])
summary_results = summary_results.round(4)

print(f"\n{'='*20} Final Summary {'='*20}")
print(f"Average scores from {n_splits}-Fold CV with {n_estimators} Trees (Criterion={criterion})")
print(summary_results)
