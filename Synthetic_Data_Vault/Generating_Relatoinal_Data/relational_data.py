import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
from sdv.sampling import Condition

# Example data for customer_data.csv
data = pd.read_csv('./data/customer_data.csv')

# Load the real data into a DataFrame
real_data = pd.DataFrame(data)

# Initialize the metadata object
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
# Initialize the synthesizer
synthesizer = CTGANSynthesizer(metadata)

# Fit the synthesizer to the real data
synthesizer.fit(real_data)

# Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=1000)

# Perform integrity checks
diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
print(diagnostic_report)

# Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
print(quality_report)

column_name = 'purchase_amount'  # Ensure this matches the actual column name
if column_name in metadata.columns:
    fig = get_column_plot(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata, column_name=column_name)
    fig.show()
else:
    print(f"Column '{column_name}' not found in the metadata.")

# Define conditions for conditional sampling
condition = Condition(num_rows=100, column_values={'purchase_amount': 250.25})

# Sample data based on conditions
conditional_synthetic_data = synthesizer.sample_from_conditions(conditions=[condition])
print(conditional_synthetic_data.head())
