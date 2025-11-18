import main as af
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COVID-19 Vaccinations & Obesity Across Ethnic Groups - Analysis Report")
print("="*80)
print()

print("Loading data...")
df1, df2 = af.load_data()
print("[OK] Data loaded successfully")
print()

print("Cleaning and merging datasets...")
df_final = af.clean_data(df1, df2)
print(f"[OK] Data cleaned successfully - Final dataset shape: {df_final.shape}")
print()

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)
print()

print("1. Vaccination Series Completion Percent Over Time by Ethnicity")
print("-" * 80)
af.exploratory_data_analysis_1(df_final)
print()

print("2. Vaccination Completion Percent by Demographic Group")
print("-" * 80)
af.exploratory_data_analysis_2(df_final)
print()

print("3. Vaccine Metrics and Obesity by Race/Ethnicity Over Time")
print("-" * 80)
af.exploratory_data_analysis_3()
print()

print("="*80)
print("MACHINE LEARNING ANALYSIS")
print("="*80)
print()

print("Calculating baseline MSE...")
baseline_mse = af.base_line_MSE(df_final)
print(f"Baseline MSE: {baseline_mse:.6f}")
print()

print("Training model and evaluating performance...")
mse_cleaned, coefficients_df, y_pred_cleaned, y_test_cleaned = af.train_and_evaluate_model(df_final)
print(f"Model MSE: {mse_cleaned:.6f}")
print()

print("Model Performance Improvement:")
improvement = ((baseline_mse - mse_cleaned) / baseline_mse) * 100
print(f"The model performs {improvement:.2f}% better than the baseline")
print()

print("Model Coefficients (sorted by absolute value):")
print("-" * 80)
print(coefficients_df.to_string())
print()

print("Generating model performance visualization...")
af.create_model_performane_graph(y_test_cleaned, y_pred_cleaned)
print()

print("="*80)
print("REPORT COMPLETE")
print("="*80)
print()
print("Key Findings:")
print("1. Black, non-Latinx group shows lowest vaccination completion rates")
print("2. Asian, non-Latinx and Other, non-Latinx groups show highest rates")
print("3. Model shows significant demographic disparities in vaccination rates")
print("4. Targeted public health initiatives recommended for underserved groups")

