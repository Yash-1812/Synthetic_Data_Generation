from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.cag import Inequality, FixedCombinations, SingleTableProgrammableConstraint
import numpy as np
from rdt.transformers.numerical import ClusterBasedNormalizer
import pandas as pd
from synth_health.ingestion import rounded_arrays

DOUBLE_LOG_COLS = ['albumin_creatinine_ratio']

"""LOG_TRANSFORM_COLS = ['serum_creatinine',
                      'albumin_creatinine_ratio]'"""
                      
DISCRETE_SNAP_COLS = {
    'bicarbonate': 0,          # integer values 15–37
    'albumin_serum': 1,        # 1-decimal values
    'calcium': 1,              # 1-decimal values
    'phosphorus': 1,           # 1-decimal values
    'uric_acid': 1,            # 1-decimal values
    'poverty_income_ratio': 1, # mostly stepped at 0.1 intervals
    'blood_urea_nitrogen': 0,  # integer
    'urine_creatinine': 0, 
    'serum_creatinine': 2,     # integer
}

PHYSIOLOGICAL_BOUNDS = {
    'age':                    (1,    120),
    'bmi':                    (10,   100),
    'weight_kg':              (5,    300),
    'height_cm':              (50,   230),
    'bp_systolic':            (60,   250),
    'bp_diastolic':           (20,   160),
    'serum_creatinine':       (0.1,  20),
    'blood_urea_nitrogen':    (1,    200),
    'albumin_serum':          (1,    6),
    'phosphorus':             (0.5,  10),
    'bicarbonate':            (5,    45),
    'calcium':                (5,    15),
    'uric_acid':              (0.5,  20),
    'urine_creatinine':       (1,    1000),
    'urine_albumin':          (0,    10000),
    'albumin_creatinine_ratio': (0,  50000),
    'poverty_income_ratio':   (0,    5),
    'education_level':        (1,    9),
}


class SyntheticDataGenerator:
    def __init__(self, schema):
        self.schema = schema
        self.model = None
        self.metadata = None
        self._real_values = {}
        self._edu_freq = None


    def _create_metadata(self, df):
        """Create metadata from dataframe"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        if 'education_level' in df.columns:
            metadata.update_column('education_level', sdtype='categorical')
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

        self.model = TVAESynthesizer(self.metadata, 
                                     epochs = 300, 
                                     embedding_dim=128, 
                                     batch_size =500,
                                     enforce_rounding = False,
                                     enforce_min_max_values = False,
                                     compress_dims=(256, 256),
                                     decompress_dims=(256, 256),
                                     l2scale= 1e-5,)
        
        self.model.auto_assign_transformers(df)

        self.model.update_transformers(column_name_to_transformer={
        'serum_creatinine': ClusterBasedNormalizer(),
        'poverty_income_ratio': ClusterBasedNormalizer()
        })
        self.model.add_constraints(constraints=[bp_constraint, diabetes_constraint, smoker_constraint, ckd_constraints])
        self.model.fit(df)

    def _store_real_values(self, df):
            """Remember the discrete value grid from real data for post-generation snapping."""
            for col in DISCRETE_SNAP_COLS.items():
                if col in df.columns:
                    decimals = DISCRETE_SNAP_COLS[col]
                    vals = df[col].dropna().round(decimals).unique()
                    self._real_values[col] = np.sort(vals)

            if 'education_level' in df.columns:
                freq = df['education_level'].dropna().value_counts(normalize=True)
                self._edu_freq = freq.sort_index()
    def _snap_to_real_values(self, synthetic_df):
        """
            For each discrete column snap every generated value to the nearest
            value seen in the real training data.  This dramatically reduces KS
            distance for heavily-peaked discrete distributions without changing
            the overall shape of the distribution.
        """
        for col, real_vals in self._real_values.items():
            if col not in synthetic_df.columns or len(real_vals) == 0:
                continue
            arr = synthetic_df[col].values.astype(float)
            # np.searchsorted gives insertion point; we compare neighbours
            idx = np.searchsorted(real_vals, arr, side='left')
            idx = np.clip(idx, 0, len(real_vals) - 1)
            # Also check the left neighbour
            idx_left = np.clip(idx - 1, 0, len(real_vals) - 1)
            left_diff = np.abs(arr - real_vals[idx_left])
            right_diff = np.abs(arr - real_vals[idx])
            snapped = np.where(left_diff < right_diff, real_vals[idx_left], real_vals[idx])
            synthetic_df[col] = snapped
        return synthetic_df
    
    def _apply_physiological_bounds(self, synthetic_df):
        """Clip hard-impossible values that can arise when enforce_min_max_values=False."""
        for col, (lo, hi) in PHYSIOLOGICAL_BOUNDS.items():
            if col in synthetic_df.columns:
                synthetic_df[col] = synthetic_df[col].clip(lower=lo, upper=hi)
        return synthetic_df
 
    def _correct_education_frequency(self, synthetic_df):
        """
        Resample education_level values to exactly match real marginal proportions.
        Only the education_level column is changed; all other columns are untouched.
        This corrects the TVAE's tendency to over-represent rare categories in the
        latent space (e.g. level 2 +6.7pp, level 9 +2.6pp above real proportions).
        """
        if self._edu_freq is None or 'education_level' not in synthetic_df.columns:
            return synthetic_df
        n = len(synthetic_df)
        corrected = np.random.choice(
            self._edu_freq.index.values,
            size=n,
            p=self._edu_freq.values
        )
        synthetic_df['education_level'] = corrected
        return synthetic_df
    
    def calculate_egfr(self, df):
        kappa = np.where(df["gender"] == "Male", 0.9, 0.7)
        alpha = np.where(df["gender"] == "Male", -0.302, -0.241)
        sex_factor = np.where(df["gender"] == "Male", 1.0, 1.012)

        serum_creatinine_ratio = df["serum_creatinine"] / kappa

        min_max = np.where(serum_creatinine_ratio <= 1, serum_creatinine_ratio ** alpha, serum_creatinine_ratio ** -1.200)
        
        df["egfr"] = round(142 * min_max * (.993**df["age"])*sex_factor,2)

        return df
    
    def rounding(self, df, col):
        amount_of_rounded_values = rounded_arrays.get(col, [2])
        df[col] = [
        round(val, np.random.choice(amount_of_rounded_values)) 
        if pd.notna(val) else val 
        for val in df[col]
    ]
        return df

    def generate(self, num_rows):
        """Generate synthetic data"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        synthetic_df = self.model.sample(num_rows)

        for col in DOUBLE_LOG_COLS:
            if col in synthetic_df.columns:
                synthetic_df[col] = np.expm1(np.expm1(synthetic_df[col]))
        
        synthetic_df = self._apply_physiological_bounds(synthetic_df)

        synthetic_df = self._snap_to_real_values(synthetic_df)

        synthetic_df = self._correct_education_frequency(synthetic_df)

        num_col_in_synth = [col for col in synthetic_df.columns if self.schema.get(col) == "numerical"]
        for col in num_col_in_synth:
            self.rounding(synthetic_df, col)

        return synthetic_df
    

        """        for col in LOG_TRANSFORM_COLS:
            if col in synthetic_df.columns:
                synthetic_df[col] = np.expm1(synthetic_df[col])"""

        """        self.calculate_egfr(synthetic_df)
        synthetic_df.loc[synthetic_df["ckd_stage"] == "Unknown", "egfr"] = np.nan

        egfr = synthetic_df['egfr']
        acr = synthetic_df['albumin_creatinine_ratio']
        conditions = [
                    (egfr >= 90) & (acr < 30),
                    (egfr >= 90) & (acr >= 30),
                    (egfr >= 60) & (egfr < 90),
                    (egfr >= 45) & (egfr < 60),
                    (egfr >= 30) & (egfr < 45),
                    (egfr >= 15) & (egfr < 30),
                    (egfr < 15),
                    (egfr.isna())
                ]
        choices = [
        'No CKD',
        'stage 1 (Kidney Damge)',
        'Stage 2 (Mildly Decreased)', 
        'Stage 3a (Mild-Moderate)', 
        'Stage 3b (Moderate-Severe)', 
        'Stage 4 (Severely Decreased)', 
        'Stage 5 (Kidney Failure)',
        'Unknown'
    ]
        
        synthetic_df['ckd_stage'] = np.select(conditions, choices, default='Unknown')
        synthetic_df['ckd_present'] = np.where(synthetic_df['ckd_stage'] == 'No CKD', 0, np.random.randint(0,2))
        synthetic_df.loc[synthetic_df["ckd_stage"] == "Unknown", "egfr"] = pd.NA"""