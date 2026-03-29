import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Evaluator:

    def __init__(self, real_df, synthetic_df, schema):
        self.real = real_df
        self.synthetic = synthetic_df
        self.schema = schema

    # -------------------------------
    # 1. KS Test (Numerical Columns)
    # -------------------------------
    def ks_test(self):
        results = {}

        for col, col_type in self.schema.items():
            if col_type == "numerical" and col in self.synthetic.columns:
                stat, p_value = ks_2samp(self.real[col], self.synthetic[col])
                results[col] = {
                    "ks_stat": stat,
                    "p_value": p_value
                }

        return results

    # -------------------------------
    # 2. Chi-Square Test (Categorical)
    # -------------------------------
    def chi_square_test(self):
        results = {}

        for col, col_type in self.schema.items():
            if col_type == "categorical" and col in self.synthetic.columns:

                real_counts = self.real[col].value_counts(normalize=True)
                synth_counts = self.synthetic[col].value_counts(normalize=True)

                # Align categories
                all_categories = set(real_counts.index).union(set(synth_counts.index))

                real_freq = []
                synth_freq = []

                for cat in all_categories:
                    real_freq.append(real_counts.get(cat, 0))
                    synth_freq.append(synth_counts.get(cat, 0))

                stat, p_value = chisquare(f_obs=synth_freq, f_exp=real_freq)

                results[col] = {
                    "chi2_stat": stat,
                    "p_value": p_value
                }

        return results

    # -------------------------------
    # 3. Correlation Difference
    # -------------------------------
    def correlation_diff(self):
        real_corr = self.real.corr(numeric_only=True)
        synth_corr = self.synthetic.corr(numeric_only=True)

        diff = (real_corr - synth_corr).abs()
        return diff

    # -------------------------------
    # 4. Classifier Test
    # -------------------------------
    def classifier_test(self):
        real = self.real.copy()
        synth = self.synthetic.copy()

        real["label"] = 0
        synth["label"] = 1

        combined = pd.concat([real, synth], ignore_index=True)

        # Drop categorical columns for simplicity
        combined = combined.select_dtypes(include=[np.number])

        X = combined.drop("label", axis=1)
        y = combined["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        return acc