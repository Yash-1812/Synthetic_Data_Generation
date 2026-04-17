import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

class DataIngestion:
    LOG_TRANSFORM_COLS = ['serum_creatinine', 'urine_albumin', 'albumin_creatinine_ratio',
                      'blood_urea_nitrogen', 'uric_acid']
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.schema = {}

    def load_data(self):
        """Load CSV into DataFrame"""
        self.df = pd.read_csv(self.file_path)
        return self.df

    def infer_schema(self):
        """Infer column types"""
        schema = {}
  
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if unique_ratio < 0.0005:
                    schema[col] = "categorical"
                else:
                    schema[col] = "numerical"

            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                schema[col] = "datetime"

            else:
                schema[col] = "text"


        self.schema = schema
        return schema

    def decimal_length(self, x):
        s = str(x)
        if '.' in s:
        # Get the part after the decimal
            decimal_part = s.split('.')[1]
            decimal_part = decimal_part.rstrip('0') # Remove trailing zeros to fix the "1.0" Pandas conversion issue

            return len(decimal_part)
    # If no decimal point, the length is 0
        return 0
    

    def rounding(self, col,mode_col):
        amount_of_rounded_values = self.df[col].apply(self.decimal_length).dropna().unique().tolist()
        #print(col,amount_of_rounded_values)
        amount_of_rounded_values = [val for val in amount_of_rounded_values if val <= 2] 
        if col in mode_col:
            amount_of_rounded_values = [1]
        #print(col,amount_of_rounded_values)
        self.df[col] = [
        round(val, np.random.choice(amount_of_rounded_values)) 
        if pd.notna(val) else val 
        for val in self.df[col]
    ]
        
    def handle_missing_values(self):
        """Basic missing value handling"""
        text_col = [col for col, t in self.schema.items() if t == "text"]
        other_col = [col for col, t in self.schema.items() if t != "text" and t != "numerical"]
        num_col = [col for col, t in self.schema.items() if t == "numerical"]

        imputer = KNNImputer(n_neighbors=20, weights="distance")
        self.df[num_col] = imputer.fit_transform(self.df[num_col])
        self.df[num_col] = self.df[num_col]
        rounded_to_1 = ['albumin_serum','phosphorus','calcium']

        for col in num_col:
            self.rounding(col, rounded_to_1)
        if 'ckd_stage' in self.df.columns:
            mask = self.df['ckd_stage'].astype(str).str.lower() == 'unknown'
            self.df.loc[mask, 'egfr'] = pd.NA
        """for col in self.LOG_TRANSFORM_COLS:
            if col in self.df.columns:
                self.df[col] = np.log1p(self.df[col])"""
        if 'age' in self.df.columns:
            self.df['age'] = self.df['age'].clip(lower=1)

    def get_basic_stats(self):
        """Return basic statistics"""
        stats = {}

        for col, col_type in self.schema.items():
            if col_type == "numerical":
                stats[col] = {
                    "mean": self.df[col].mean(),
                    "std": self.df[col].std(),
                    "min": self.df[col].min(),
                    "max": self.df[col].max()
                }

            elif col_type == "categorical":
                stats[col] = {
                    "unique_values": self.df[col].nunique(),
                    "top": self.df[col].mode()[0]
                }

        return stats
    
    def process(self):
        self.load_data()
        self.infer_schema()
        self.handle_missing_values()
        stats = self.get_basic_stats()
        return self.df, self.schema, stats