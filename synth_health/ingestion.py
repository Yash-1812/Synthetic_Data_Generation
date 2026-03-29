import pandas as pd

class DataIngestion:
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
            if pd.api.types.is_numeric_dtype(self.df[col]):
                schema[col] = "numerical"

            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                schema[col] = "datetime"

            else:
                # Heuristic: low unique values → categorical
                unique_ratio = self.df[col].nunique() / len(self.df)

                if unique_ratio < 0.05:
                    schema[col] = "categorical"
                else:
                    schema[col] = "text"

        self.schema = schema
        return schema

    def handle_missing_values(self):
        """Basic missing value handling"""
        for col, col_type in self.schema.items():

            if col_type == "numerical":
                self.df[col] = self.df[col].apply(lambda x: x if x > 1e-6 else pd.NA)
                self.df[col] = self.df[col].fillna(self.df[col].mean())

            elif col_type == "categorical":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

            else:
                self.df[col] = self.df[col].fillna("unknown")

        return self.df

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
        """Run full pipeline"""
        self.load_data()
        self.infer_schema()
        self.handle_missing_values()
        stats = self.get_basic_stats()

        return self.df, self.schema, stats