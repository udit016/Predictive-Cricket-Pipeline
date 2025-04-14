from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

class BlendedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_params=None, lgbm_params=None, cat_params=None, gbm_params=None, cv=5, scoring='accuracy'):
        self.xgb_params = xgb_params if xgb_params is not None else {}
        self.lgbm_params = lgbm_params if lgbm_params is not None else {}
        self.cat_params = cat_params if cat_params is not None else {}
        self.gbm_params = gbm_params if gbm_params is not None else {}

        self.cv = cv
        self.scoring = scoring

        self.model_xgb = XGBClassifier(**self.xgb_params)
        self.model_lgbm = LGBMClassifier(**self.lgbm_params)
        self.cat_params.setdefault("verbose", 0)
        self.model_cat = CatBoostClassifier(**self.cat_params)
        self.model_gbm = GradientBoostingClassifier(**self.gbm_params)

        self.xgb_wt = 1.0
        self.lgbm_wt = 1.0
        self.cat_wt = 1.0
        self.gbm_wt = 1.0

    def _compute_weights(self, X, y):
        print("🔄 Computing model cross-validation scores...")

        scores = {}
        scores['xgb'] = np.mean(cross_val_score(self.model_xgb, X, y, cv=self.cv, scoring=self.scoring))
        scores['lgbm'] = np.mean(cross_val_score(self.model_lgbm, X, y, cv=self.cv, scoring=self.scoring))
        scores['cat'] = np.mean(cross_val_score(self.model_cat, X, y, cv=self.cv, scoring=self.scoring))
        scores['gbm'] = np.mean(cross_val_score(self.model_gbm, X, y, cv=self.cv, scoring=self.scoring))

        print(f"✅ CV Scores: {scores}")

        total = sum(scores.values())
        self.xgb_wt = scores['xgb'] / total
        self.lgbm_wt = scores['lgbm'] / total
        self.cat_wt = scores['cat'] / total
        self.gbm_wt = scores['gbm'] / total

        print(f"✅ Normalized Weights: xgb={self.xgb_wt:.2f}, lgbm={self.lgbm_wt:.2f}, cat={self.cat_wt:.2f}, gbm={self.gbm_wt:.2f}")

    def fit(self, X, y):
        # Compute weights
        self._compute_weights(X, y)

        # Train all models on full training data
        self.model_xgb.fit(X, y)
        self.model_lgbm.fit(X, y)
        self.model_cat.fit(X, y)
        self.model_gbm.fit(X, y)

        return self

    def predict_proba(self, X):
        proba_xgb = self.model_xgb.predict_proba(X)
        proba_lgbm = self.model_lgbm.predict_proba(X)
        proba_cat = self.model_cat.predict_proba(X)
        proba_gbm = self.model_gbm.predict_proba(X)

        blended_proba = (
            self.xgb_wt * proba_xgb +
            self.lgbm_wt * proba_lgbm +
            self.cat_wt * proba_cat +
            self.gbm_wt * proba_gbm
        )

        return blended_proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def get_params(self, deep=True):
        return {
            "xgb_params": self.xgb_params,
            "lgbm_params": self.lgbm_params,
            "cat_params": self.cat_params,
            "gbm_params": self.gbm_params,
            "cv": self.cv,
            "scoring": self.scoring
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)

        self.model_xgb = XGBClassifier(**self.xgb_params)
        self.model_lgbm = LGBMClassifier(**self.lgbm_params)
        self.cat_params.setdefault("verbose", 0)
        self.model_cat = CatBoostClassifier(**self.cat_params)
        self.model_gbm = GradientBoostingClassifier(**self.gbm_params)

        return self