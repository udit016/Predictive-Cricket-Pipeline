import logging
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from src.model_building import BlendedModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def optimize_hyperparameters(X_train, y_train, cv_folds=3, n_trials=50):
    """
    Optimizes hyperparameters for the BlendedModel using Optuna.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cv_folds (int): Number of cross-validation folds.
        n_trials (int): Number of Optuna trials.

    Returns:
        best_model: The trained blended model with best-found hyperparameters.
        best_params: Dictionary of the best hyperparameters.
    """
    def objective(trial):
        # XGBoost parameters
        xgb_params = {
            'max_depth': trial.suggest_int("xgb_max_depth", 2, 8),
            'learning_rate': trial.suggest_loguniform("xgb_learning_rate", 0.01, 0.1),
            'colsample_bytree': trial.suggest_loguniform("xgb_colsample_bytree", 0.1, 0.8),
            'n_estimators': trial.suggest_int("xgb_n_estimators", 50, 200),
            'reg_alpha': trial.suggest_loguniform("xgb_reg_alpha", 0.5, 1.5),
            'reg_lambda': trial.suggest_loguniform("xgb_reg_lambda", 0.5, 1.7),
            'gamma': trial.suggest_loguniform("xgb_gamma", 0.1, 0.7),
        }

        # LightGBM parameters
        lgbm_params = {
            'n_estimators': trial.suggest_int("lgbm_n_estimators", 30, 200),
            'num_leaves': trial.suggest_int("lgbm_num_leaves", 20, 100),
            'bagging_fraction': trial.suggest_loguniform("lgbm_bagging_fraction", 0.1, 0.7),
            'bagging_freq': trial.suggest_int("lgbm_bagging_freq", 1, 7),
            'learning_rate': trial.suggest_loguniform("lgbm_learning_rate", 0.01, 0.1),
            'min_data_in_leaf': trial.suggest_int("lgbm_min_data_in_leaf", 1, 7),
        }

        # CatBoost parameters
        cat_params = {
            'iterations': trial.suggest_int("cat_iterations", 50, 500),
            'depth': trial.suggest_int("cat_depth", 3, 8),
            'learning_rate': trial.suggest_loguniform("cat_learning_rate", 0.001, 0.1),
            'l2_leaf_reg': trial.suggest_loguniform("cat_l2_leaf_reg", 0.5, 1.5),
        }

        # GradientBoostingClassifier (GBM) parameters
        gbm_params = {
            'n_estimators': trial.suggest_int("gbm_n_estimators", 50, 200),
            'max_depth': trial.suggest_int("gbm_max_depth", 2, 10),
            'learning_rate': trial.suggest_loguniform("gbm_learning_rate", 0.01, 0.2),
            'subsample': trial.suggest_loguniform("gbm_subsample", 0.5, 1.0),
            'min_samples_split': trial.suggest_int("gbm_min_samples_split", 2, 10),
            'min_samples_leaf': trial.suggest_int("gbm_min_samples_leaf", 1, 5),
            'max_features': trial.suggest_categorical("gbm_max_features", ["sqrt", "log2", None]),
        }

        model = BlendedModel(
            xgb_params=xgb_params,
            lgbm_params=lgbm_params,
            cat_params=cat_params,
            gbm_params=gbm_params
        )

        f1_scorer = make_scorer(f1_score, zero_division=0)
        cv_scores = cross_val_score(model, X_train, y_train, scoring=f1_scorer, cv=cv_folds)
        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    logging.info("Best trial value: {:.4f}".format(study.best_trial.value))
    logging.info("Best hyperparameters:")
    for key, value in best_params.items():
        logging.info(f"  {key}: {value}")

    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Extract best param sets
    best_xgb_params = {
        'max_depth': best_params["xgb_max_depth"],
        'learning_rate': best_params["xgb_learning_rate"],
        'colsample_bytree': best_params["xgb_colsample_bytree"],
        'n_estimators': best_params["xgb_n_estimators"],
        'reg_alpha': best_params["xgb_reg_alpha"],
        'reg_lambda': best_params["xgb_reg_lambda"],
        'gamma': best_params["xgb_gamma"],
    }
    best_lgbm_params = {
        'n_estimators': best_params["lgbm_n_estimators"],
        'num_leaves': best_params["lgbm_num_leaves"],
        'bagging_fraction': best_params["lgbm_bagging_fraction"],
        'bagging_freq': best_params["lgbm_bagging_freq"],
        'learning_rate': best_params["lgbm_learning_rate"],
        'min_data_in_leaf': best_params["lgbm_min_data_in_leaf"],
    }
    best_cat_params = {
        'iterations': best_params["cat_iterations"],
        'depth': best_params["cat_depth"],
        'learning_rate': best_params["cat_learning_rate"],
        'l2_leaf_reg': best_params["cat_l2_leaf_reg"],
    }
    best_gbm_params = {
        'n_estimators': best_params["gbm_n_estimators"],
        'max_depth': best_params["gbm_max_depth"],
        'learning_rate': best_params["gbm_learning_rate"],
        'subsample': best_params["gbm_subsample"],
        'min_samples_split': best_params["gbm_min_samples_split"],
        'min_samples_leaf': best_params["gbm_min_samples_leaf"],
        'max_features': best_params["gbm_max_features"]
    }

    best_model = BlendedModel(
        xgb_params=best_xgb_params,
        lgbm_params=best_lgbm_params,
        cat_params=best_cat_params,
        gbm_params=best_gbm_params
    )

    best_model.fit(X_train, y_train)
    return best_model, best_params