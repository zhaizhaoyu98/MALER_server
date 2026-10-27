"""Preserve user labels with the retained XGBoost 1.6 classification API."""
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class LabelledXGBClassifier(XGBClassifier):
    """Learn the label mapping only from the current training fold."""

    def fit(self, X, y, **fit_params):
        encoder = LabelEncoder().fit(y)
        encoded = encoder.transform(y)
        if fit_params.get('eval_set') is not None:
            fit_params['eval_set'] = [(values, encoder.transform(labels))
                                      for values, labels in fit_params['eval_set']]
        super(LabelledXGBClassifier, self).fit(X, encoded, **fit_params)
        self.label_encoder_ = encoder
        self.classes_ = encoder.classes_.copy()
        return self

    def predict(self, X, output_margin=False, **predict_params):
        prediction = super(LabelledXGBClassifier, self).predict(
            X, output_margin=output_margin, **predict_params)
        if output_margin:
            return prediction
        return self.label_encoder_.inverse_transform(prediction.astype(int))
