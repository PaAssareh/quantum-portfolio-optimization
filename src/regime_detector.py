from hmmlearn import hmm


class RegimeDetector:
    def __init__(self, feature_df, model="hmm", states=3, threshold=0.7):
        self.feature_df = feature_df
        self.model = model
        self.states = states
        self.threshold = threshold
        self.regimes = None

    def detect_regimes(self):
        if self.model != "hmm":
            raise NotImplementedError(f"Model {self.model} not implemented in phase 1")

        X = self.feature_df.values.astype(float)
        model = hmm.GaussianHMM(
            n_components=self.states,
            covariance_type="full",
            n_iter=200,
            random_state=42
        )
        model.fit(X)
        self.regimes = model.predict(X)
        return self.regimes

    def get_current_regime(self):
        if self.regimes is None or len(self.regimes) == 0:
            return None
        return int(self.regimes[-1])