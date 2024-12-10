import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_year_data(file_path, year):
    # Read CSV with thousands separator
    df = pd.read_csv(file_path, thousands=',')

    # Convert string values to numeric, replacing '-' with 0
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].replace('-', '0'), errors='coerce')

    # Get the Total rows
    total_row = df.iloc[0:3]

    total_under_12 = total_row.iloc[:,2].fillna(0).sum() + total_row.iloc[:,5].fillna(0).sum()
    total_12_15 = total_row.iloc[:,3].fillna(0).sum() + total_row.iloc[:,6].fillna(0).sum()
    total_15_18 = total_row.iloc[:,4].fillna(0).sum() + total_row.iloc[:,7].fillna(0).sum()

    male_under_12 = total_row.iloc[1,2] + total_row.iloc[1,5]
    male_12_15 = total_row.iloc[1,3] + total_row.iloc[1,6]
    male_15_18 = total_row.iloc[1,4] + total_row.iloc[1,7]

    female_under_12 = total_row.iloc[2,2] + total_row.iloc[2,5]
    female_12_15 = total_row.iloc[2,3] + total_row.iloc[2,6]
    female_15_18 = total_row.iloc[2,4] + total_row.iloc[2,7]

    return {
        'Year': year,
        'Total_Under_12': total_under_12,
        'Total_12_15': total_12_15,
        'Total_15_18': total_15_18,
        'Male_Under_12': male_under_12,
        'Male_12_15': male_12_15,
        'Male_15_18': male_15_18,
        'Female_Under_12': female_under_12,
        'Female_12_15': female_12_15,
        'Female_15_18': female_15_18
    }

# Process all years
years = range(2017, 2024)
data = []
for year in years:
    # Update file path according to your directory structure
    file_path = f'Overview of Victims Reported in Child and Youth Sexual Exploitation Cases - {year}.csv'
    try:
        data.append(process_year_data(file_path, year))
    except FileNotFoundError:
        print(f"File not found for year {year}")
        continue

# Create DataFrame
df_combined = pd.DataFrame(data)

# Create visualizations


# 1. Total Cases by Year
plt.figure(figsize=(12, 6))
yearly_total = df_combined[['Year', 'Total_Under_12', 'Total_12_15', 'Total_15_18']].set_index('Year')
yearly_total.plot(kind='bar', stacked=True)
plt.title('Total Cases by Year and Age Group')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.legend(title='Age Group', labels=['Under 12', '12-15', '15-18'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Gender Distribution Over Years
plt.figure(figsize=(12, 6))
male_total = df_combined[['Male_Under_12', 'Male_12_15', 'Male_15_18']].sum(axis=1)
female_total = df_combined[['Female_Under_12', 'Female_12_15', 'Female_15_18']].sum(axis=1)
plt.bar(df_combined['Year'], male_total, label='Male')
plt.bar(df_combined['Year'], female_total, bottom=male_total, label='Female')
plt.title('Gender Distribution Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Age Group Distribution
plt.figure(figsize=(12, 6))
age_groups = ['Under 12', '12-15', '15-18']
age_totals = [
    df_combined['Total_Under_12'].sum(),
    df_combined['Total_12_15'].sum(),
    df_combined['Total_15_18'].sum()
]
plt.pie(age_totals, labels=age_groups, autopct='%1.1f%%')
plt.title('Distribution by Age Group (2017-2023)')
plt.axis('equal')
plt.show()
