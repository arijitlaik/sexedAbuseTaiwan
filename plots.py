# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure plots render correctly in Jupyter Notebook (if using one)
# %matplotlib inline

# Increase the default figure size
plt.rcParams['figure.figsize'] = (12, 8)

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('clean_data.csv')

# Display the first few rows of the data
print("Initial Data:")
print(data.head())

# Data Cleaning

# Replace '-' with NaN and convert numerical columns to integers
data.replace('-', np.nan, inplace=True)

# List of columns to convert
numeric_cols = ['General Under 12 Years', 'General 12-Under 15 Years', 'General 15-Under 18 Years',
                'Indigenous Under 12 Years', 'Indigenous 12-Under 15 Years', 'Indigenous 15-Under 18 Years']

# Convert columns to numeric types (int), coerce errors to handle NaN
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill NaN values with 0, as missing counts imply zero occurrences
data[numeric_cols] = data[numeric_cols].fillna(0).astype(int)

# Verify data types after conversion
print("\nData Types after Cleaning:")
print(data.dtypes)

# Combine Counts

# Create total counts for each age group by adding 'General' and 'Indigenous' counts
data['Under 12 Years'] = data['General Under 12 Years'] + data['Indigenous Under 12 Years']
data['12-Under 15 Years'] = data['General 12-Under 15 Years'] + data['Indigenous 12-Under 15 Years']
data['15-Under 18 Years'] = data['General 15-Under 18 Years'] + data['Indigenous 15-Under 18 Years']

# Drop the 'General' and 'Indigenous' columns as they are no longer needed
data.drop(columns=numeric_cols, inplace=True)

# Display the cleaned and combined data
print("\nCleaned and Combined Data:")
print(data.head())

# Data Aggregation

# Melt the DataFrame for easier plotting
melted_data = data.melt(id_vars=['Year', 'Region', 'Gender'],
                        value_vars=['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years'],
                        var_name='Age Group', value_name='Count')

# Verify the melted data
print("\nMelted Data:")
print(melted_data.head())

# Visualization

# Yearly Trend from 2017 to 2023 by Age Group

# Aggregate data by Year and Age Group
yearly_age_data = melted_data.groupby(['Year', 'Age Group'])['Count'].sum().reset_index()

# Plot the yearly trend for each age group
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_age_data, x='Year', y='Count', hue='Age Group', marker='o')
plt.title('Yearly Trend by Age Group (2017-2023)')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Age Group')
plt.xticks(sorted(data['Year'].unique()))
plt.grid(True)
plt.show()

# Yearly Trend by Gender and Age Group

# Aggregate data by Year, Gender, and Age Group
yearly_gender_age_data = melted_data.groupby(['Year', 'Gender', 'Age Group'])['Count'].sum().reset_index()

# Plot the yearly trend by Gender and Age Group
plt.figure(figsize=(12, 8))
sns.lineplot(data=yearly_gender_age_data, x='Year', y='Count', hue='Gender', style='Age Group', markers=True, dashes=False)
plt.title('Yearly Trend by Gender and Age Group (2017-2023)')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Gender and Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(sorted(data['Year'].unique()))
plt.grid(True)
plt.show()

# Yearly Trend for Top 5 Regions

# Identify the top 5 regions with the highest total counts over the years
top_regions = data.groupby('Region')[['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']].sum().sum(axis=1).nlargest(5).index

# Filter data for top regions
top_regions_data = data[data['Region'].isin(top_regions)]

# Melt the top regions data for plotting
top_regions_melted = top_regions_data.melt(id_vars=['Year', 'Region', 'Gender'],
                                           value_vars=['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years'],
                                           var_name='Age Group', value_name='Count')

# Aggregate by Year and Region
yearly_region_data = top_regions_melted.groupby(['Year', 'Region', 'Age Group'])['Count'].sum().reset_index()

# Plotting the yearly trend for top regions
plt.figure(figsize=(12, 8))
sns.lineplot(data=yearly_region_data, x='Year', y='Count', hue='Region', style='Age Group', markers=True)
plt.title('Yearly Trend for Top 5 Regions (2017-2023)')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Region and Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(sorted(data['Year'].unique()))
plt.grid(True)
plt.show()
