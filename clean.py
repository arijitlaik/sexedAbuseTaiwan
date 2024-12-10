import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read and clean data
def clean_and_save_data():
    # Read original data
    df = pd.read_csv('All Years.csv')
    df['Region'] = df['Region'].fillna(method='ffill')

    # Create clean dataframe excluding totals
    clean_df = df[(df['Region'] != 'Total') & (df['Gender'] != 'Total')].copy()

    # Save clean data to CSV
    clean_df.to_csv('clean_data.csv', index=False)
    print("Clean data saved to 'clean_data.csv'")

    return clean_df
