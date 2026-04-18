from synth_health.ingestion import DataIngestion
from synth_health.model import SyntheticDataGenerator
from synth_health.evaluate import Evaluator
import numpy as np
import os


DOUBLE_LOG_COLS = ['albumin_creatinine_ratio']

def naming_function(df, base_name):
    """
    Saves df to the first available filename (e.g., processed_data2.csv)
    """
    n = 1
    # Determine the directory/prefix based on your input
    if base_name == "processed_data":
        folder = "processed_data/"
        prefix = "processed_data"
    else:
        folder = "synthetic_output/"
        prefix = "synthetic_output"
        # Create folder if it doesn't exist to avoid FileNotFoundError
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 1. Loop to find the first number that DOES NOT exist yet
    while os.path.exists(f"{folder}{prefix}{n}.csv"):
        n += 1

    # 2. Save the file once
    filename = f"{folder}{prefix}{n}.csv"
    df.to_csv(filename, index=False)
    
    print(f"Successfully saved: {filename}")
    return filename

# Step 1: Load data
ingestor = DataIngestion("data/CKD_NHANES_2021_2023.csv")
df, schema, stats = ingestor.process()

# Drop ID column
df = df.drop(columns=["participant_id", "egfr"])
print("Schema:", schema)
naming_function(df, "processed_data")
# Step 2: Train model
generator = SyntheticDataGenerator(schema)
generator.train(df)

# Step 3: Generate synthetic data
synthetic_df = generator.generate(8000)

real_df_for_eval = df.copy()
for col in DOUBLE_LOG_COLS:
    if col in real_df_for_eval.columns:
        # Invert double-log: expm1 twice
        real_df_for_eval[col] = np.expm1(np.expm1(real_df_for_eval[col]))

naming_function(synthetic_df, "synthetic_output")

print("\nSynthetic Data Sample:")
print(synthetic_df.head())

# Evaluation of the model
evaluator = Evaluator(real_df_for_eval, synthetic_df, schema)

ks_results = evaluator.ks_test()
chi_results = evaluator.chi_square_test()
corr_diff = evaluator.correlation_diff()
clf_acc = evaluator.classifier_test()

print("\nKS Test Results:", ks_results)
print("\nChi-Square Results:", chi_results)
print("\nClassifier Accuracy (Real vs Synthetic):", clf_acc)