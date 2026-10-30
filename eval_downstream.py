import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error, mean_absolute_error, r2_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

def evaluate_downstream(dataname, variant='D', ratio='30', mask_type='MCAR', num_trials=10, num_steps=50):
    info_path = os.path.join(DATA_DIR, 'Info', f'{dataname}.json')
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    task_type = info.get('task_type', 'binclass') # 'binclass', 'multiclass', atau 'regression'

    variant_suffix = '' if variant == 'D' else f'_{variant}'
    mask_folder = 'MNAR_logistic_T2' if mask_type == 'MNAR' else mask_type

    pattern = os.path.join(ROOT_DIR, 'results', dataname, f'rate{ratio}', mask_folder, '*', f'{num_trials}_{num_steps}{variant_suffix}')
    split_dirs = sorted(glob.glob(pattern))

    rf_metrics_1, rf_metrics_2 = [], []
    gb_metrics_1, gb_metrics_2 = [], []

    for sdir in split_dirs:
        train_csv = os.path.join(sdir, 'train_imputed.csv')
        test_csv = os.path.join(sdir, 'test_imputed.csv')
        if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
            continue

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        cols = train_df.columns

        # Pisahkan fitur (X) dan target (y) berdasarkan indeks kolom di Info JSON
        feature_idx = num_col_idx + cat_col_idx
        target_col = cols[target_col_idx[0]]

        X_train = train_df[cols[feature_idx]].copy()
        y_train = train_df[target_col].values
        X_test = test_df[cols[feature_idx]].copy()
        y_test = test_df[target_col].values

        # Encode fitur kategorikal jika ada
        if len(cat_col_idx) > 0:
            cat_cols = cols[cat_col_idx]
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
            X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float32))
        X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

        if task_type in ('binclass', 'multiclass') or train_df[target_col].dtype == object:
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train.astype(str))
            y_test_enc = le.transform(y_test.astype(str))

            # 1. Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train_enc)
            pred_rf = rf.predict(X_test_scaled)
            rf_metrics_1.append(accuracy_score(y_test_enc, pred_rf) * 100)
            rf_metrics_2.append(f1_score(y_test_enc, pred_rf, average='macro', zero_division=0) * 100)

            # 2. Gradient Boosting Classifier
            gb = GradientBoostingClassifier(random_state=42)
            gb.fit(X_train_scaled, y_train_enc)
            pred_gb = gb.predict(X_test_scaled)
            gb_metrics_1.append(accuracy_score(y_test_enc, pred_gb) * 100)
            gb_metrics_2.append(f1_score(y_test_enc, pred_gb, average='macro', zero_division=0) * 100)
            is_clf = True
        else:
            y_train_flt = y_train.astype(np.float32)
            y_test_flt = y_test.astype(np.float32)

            # 1. Random Forest Regressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train_flt)
            pred_rf = rf.predict(X_test_scaled)
            rf_metrics_1.append(root_mean_squared_error(y_test_flt, pred_rf))
            rf_metrics_2.append(r2_score(y_test_flt, pred_rf) * 100)

            # 2. Gradient Boosting Regressor
            gb = GradientBoostingRegressor(random_state=42)
            gb.fit(X_train_scaled, y_train_flt)
            pred_gb = gb.predict(X_test_scaled)
            gb_metrics_1.append(root_mean_squared_error(y_test_flt, pred_gb))
            gb_metrics_2.append(r2_score(y_test_flt, pred_gb) * 100)
            is_clf = False

    if not rf_metrics_1:
        print(f"Belum ditemukan file train_imputed.csv / test_imputed.csv untuk {dataname} ({variant}).")
        return

    print("=" * 65)
    print(f"HASIL DOWNSTREAM TASK ({len(rf_metrics_1)} Masks) | Dataset: {dataname.upper()} | Variant: {variant}")
    print("=" * 65)
    if is_clf:
        print(f"[Random Forest Classifier]")
        print(f"  Test Accuracy : {np.mean(rf_metrics_1):.2f}% ± {np.std(rf_metrics_1):.2f}%")
        print(f"  Test Macro-F1 : {np.mean(rf_metrics_2):.2f}% ± {np.std(rf_metrics_2):.2f}%")
        print(f"[Gradient Boosting Classifier]")
        print(f"  Test Accuracy : {np.mean(gb_metrics_1):.2f}% ± {np.std(gb_metrics_1):.2f}%")
        print(f"  Test Macro-F1 : {np.mean(gb_metrics_2):.2f}% ± {np.std(gb_metrics_2):.2f}%")
    else:
        print(f"[Random Forest Regressor]")
        print(f"  Test RMSE     : {np.mean(rf_metrics_1):.4f} ± {np.std(rf_metrics_1):.4f}")
        print(f"  Test R2 Score : {np.mean(rf_metrics_2):.2f}% ± {np.std(rf_metrics_2):.2f}%")
        print(f"[Gradient Boosting Regressor]")
        print(f"  Test RMSE     : {np.mean(gb_metrics_1):.4f} ± {np.std(gb_metrics_1):.4f}")
        print(f"  Test R2 Score : {np.mean(gb_metrics_2):.2f}% ± {np.std(gb_metrics_2):.2f}%")
    print("=" * 65)

if __name__ == '__main__':
    # Contoh pemanggilan langsung
    evaluate_downstream(dataname='shoppers', variant='D')