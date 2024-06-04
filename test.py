import pandas as pd

# Sample DataFrame
data = {
    'colsample_bylevel': [0.635420, 0.021278, 0.277792, 0.901090, 0.022860],
    'colsample_bytree': [0.014154, 0.569158, 0.281579, 0.844287, 0.844287],
    'eta_log': [-3, -8, -1, -7, -7],
    'max_depth': [4, 12, 6, 9, 1],
    'min_child_weight_log': [3, 7, 4, 1, 1],
    'reg_alpha_log': [-5, 4, 7, -6, -6],
    'reg_lambda_log': [-10, 6, -4, 1, 5],
    'subsample_per_it': [0.809736, 0.509225, 0.177123, 0.692058, 0.727550]
}

df = pd.DataFrame(data)

# Define legal ranges as a tuple
legal_ranges = ((0., 1.), (0., 1.),(-10.0, -0.0), (1, 15), (0.0, 10.0), (-10, 5.0), (-10, 10.0), (0., 1.))

# List of columns in the same order as the legal ranges
columns = [
    'colsample_bylevel',
    'colsample_bytree',
    'eta_log',
    'max_depth',
    'min_child_weight_log',
    'reg_alpha_log',
    'reg_lambda_log',
    'subsample_per_it'
]

# Function to check if a value is within the legal range
def in_legal_range(value, legal_range):
    return legal_range[0] <= value <= legal_range[1]

# Get the indices of rows where all values are within their legal ranges
legal_row=df[df.apply(lambda row: all(in_legal_range(row[col], legal_ranges[i]) for i, col in enumerate(columns)), axis=1)]
legal_row_indices = legal_row.index
print(legal_row)
import numpy as np
np_array = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
    [130, 140, 150, 160],
    [170, 180, 190, 200]
])

# Use legal_row_indices to get the slice of the NumPy array
filtered_np_array = np_array[legal_row_indices]

print("Legal Row Indices:", legal_row_indices.tolist())
print("Filtered NumPy Array:\n", type(filtered_np_array))
