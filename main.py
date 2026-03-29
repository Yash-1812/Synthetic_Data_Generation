from synth_health.ingestion import DataIngestion
from synth_health.model import SyntheticDataGenerator
from synth_health.evaluate import Evaluator

# Step 1: Load data
ingestor = DataIngestion("data/CKD_NHANES_2021_2023.csv")
df, schema, stats = ingestor.process()

# Drop ID column
df = df.drop(columns=["participant_id"])

print("Schema:", schema)

# Step 2: Train model
generator = SyntheticDataGenerator(schema)
generator.train(df)

# Step 3: Generate synthetic data
synthetic_df = generator.generate(1000)

print("\nSynthetic Data Sample:")
print(synthetic_df.head())

# Evaluation of the model

evaluator = Evaluator(df, synthetic_df, schema)

ks_results = evaluator.ks_test()
chi_results = evaluator.chi_square_test()
corr_diff = evaluator.correlation_diff()
clf_acc = evaluator.classifier_test()

print("\nKS Test Results:", ks_results)
print("\nChi-Square Results:", chi_results)
print("\nClassifier Accuracy (Real vs Synthetic):", clf_acc)