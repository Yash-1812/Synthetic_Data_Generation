from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.cag import Inequality, FixedCombinations
import numpy as np

LOG_TRANSFORM_COLS = ['serum_creatinine', 'urine_albumin', 'albumin_creatinine_ratio',
                      'blood_urea_nitrogen', 'uric_acid']
class SyntheticDataGenerator:
    def __init__(self, schema):
        self.schema = schema
        self.model = None
        self.metadata = None

    def _create_metadata(self, df):
        """Create metadata from dataframe"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        return metadata

    def train(self, df):
        """Train CTGAN model"""
        self.metadata = self._create_metadata(df)
        ckd_constraints = FixedCombinations(
            column_names=['ckd_present', 'ckd_stage'])
        diabetes_constraint = FixedCombinations(
            column_names=['diabetes_diagnosed', 'insulin_use', 'diabetes_pills'])
        smoker_constraint = FixedCombinations(
            column_names=['ever_smoked', 'current_smoker'])
        bp_constraint = Inequality(
            low_column_name='bp_diastolic',
            high_column_name='bp_systolic'
        )

        self.model = CTGANSynthesizer(self.metadata, epochs = 500, batch_size =500,enforce_rounding = True, enforce_min_max_values = True)
        self.model.add_constraints(constraints=[bp_constraint, diabetes_constraint, smoker_constraint, ckd_constraints])
        self.model.fit(df)

    def generate(self, num_rows):
        """Generate synthetic data"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        synthetic_df = self.model.sample(num_rows)
        """ for col in LOG_TRANSFORM_COLS:
            if col in synthetic_df.columns:
                synthetic_df[col] = np.expm1(synthetic_df[col])"""

        return synthetic_df