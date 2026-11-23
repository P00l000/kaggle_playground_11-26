from datetime import datetime
import os
from pathlib import Path

import os
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
)
from utils.config import EXPORT_DIR
# from utils.env import TABPFN_TOKEN


def log_run(row: dict, registry_path: Path):
    """Add or replace a row in the registry CSV (one row per model name)."""
    new = pd.DataFrame([row])
    if registry_path.exists():
        old = pd.read_csv(registry_path)
        old = old[old["model_name"] != row["model_name"]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(registry_path, index=False)


def run_experiment(name, spec_model, X_train, y_train, X_test, test_df,
                   id_column, models_dir, data_dir, cv=5):
    # os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN
    model = spec_model["model"]
    param_grid = spec_model.get("param_grid")

    tabpfn_config = spec_model.get("tabpfn")

    if tabpfn_config:
        max_rows = tabpfn_config.get("max_training_rows", 10_000)
        max_features = tabpfn_config.get("max_features", 500)

        if len(X_train) > max_rows:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=max_rows,
                random_state=42,
            )
            subset_idx, _ = next(splitter.split(X_train, y_train))
            X_train = X_train.iloc[subset_idx]
            y_train = y_train.iloc[subset_idx]

        if X_train.shape[1] > max_features:
            variances = X_train.var(numeric_only=True)
            retained_columns = variances.nlargest(max_features).index
            X_train = X_train.loc[:, retained_columns]
            X_test = X_test.loc[:, retained_columns]
    
    if param_grid:
        # grid search: picks the best config, then refits it on the whole X_train
        search = GridSearchCV(
    model,
    param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    pre_dispatch=1,
    refit=True,
)
        search.fit(X_train, y_train)

        final_model = search.best_estimator_
        best_params = search.best_params_
        val_mean = search.best_score_
        val_std = search.cv_results_["std_test_score"][search.best_index_]
    else:
        # no grid: plain cross-validation, then refit on the whole X_train
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=1)
        final_model = clone(model).fit(X_train, y_train)
        best_params = {}
        val_mean, val_std = scores.mean(), scores.std()

    # save the weights 
    weights_path = Path(models_dir) / f"{name}.joblib"
    models_dir.mkdir(parents=True, exist_ok=True) 
    joblib.dump(final_model, weights_path)

    # predict on the test set
    test_proba = final_model.predict_proba(X_test)[:, 1]
    predictions_path = Path(EXPORT_DIR) / f"{name}_predictions.csv"
    pd.DataFrame({id_column: test_df[id_column],
                  "is_fraud": test_proba}).to_csv(predictions_path, index=False)

    log_run(
        {
            "model_name": name,
            "model_type": spec_model["type"],
            "weights_path": str(weights_path),
            "predictions_path": str(predictions_path),
            "val_score_mean": val_mean,
            "val_score_std": val_std,
            "best_params": str(best_params),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        registry_path=Path(EXPORT_DIR) / "model_registry.csv",
    )
    print(f"{name}: ROC-AUC = {val_mean:.4f} ± {val_std:.4f}")
    if best_params:
        print(f"{name}: best params = {best_params}")

    return final_model