# Import necessary libraries
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

# Example data for customer_data.csv
# data = pd.read_csv('./data/customer_data.csv')
data = pd.read_csv(r"C:\Users\Admin\Documents\GEN_AI\SDV\Eval_Validation_SyntheticData\customer_data.csv")

# Load the real data into a DataFrame
real_data = pd.DataFrame(data)

# Initialize the metadata object
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Initialize a model using CTGANSynthesizer
model = CTGANSynthesizer(metadata)

# Fit the model on the real data
model.fit(data)

# Generate synthetic data
synthetic_data = model.sample(num_rows=10)

# Convert metadata to dictionary
metadata_dict = metadata.to_dict()

# Evaluate the synthetic data quality
report = QualityReport()
report.generate(real_data=data, synthetic_data=synthetic_data, metadata=metadata_dict)

# Display synthetic data and quality report
print("Synthetic Data:")
print(synthetic_data)
print("\nQuality Score:")
print(report.get_score())