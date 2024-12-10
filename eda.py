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
    file_path = f'taiwan/Overview of Victims Reported in Child and Youth Sexual Exploitation Cases - {year}.csv'
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
plt.savefig('total_cases_by_year.png')
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
plt.savefig('gender_distribution_over_years.png')
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
plt.savefig('age_group_distribution.png')
plt.show()

# 4. Regional Distribution
def process_region_data(file_path, year):
    # Read CSV with thousands separator
    df = pd.read_csv(file_path, thousands=',')

    # Convert string values to numeric, replacing '-' with 0
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].replace('-', '0'), errors='coerce')

    # Drop the first row (Total) and the empty columns
    df = df.drop([0, 1, 2]).dropna(axis=1, how='all')

    # Ensure the first column is treated as strings
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)

    # Extract region names
    df['Region'] = df.iloc[:, 0].str.split(',').str[0]

    # Sum the values for General and Indigenous
    df['Under_12'] = df.iloc[:, 2] + df.iloc[:, 5]
    df['12_15'] = df.iloc[:, 3] + df.iloc[:, 6]
    df['15_18'] = df.iloc[:, 4] + df.iloc[:, 7]

    # Add the year column
    df['Year'] = year

    return df[['Year', 'Region', 'Under_12', '12_15', '15_18']]

# Process all years for regional data
region_data = []
for year in years:
    file_path = f'taiwan/Overview of Victims Reported in Child and Youth Sexual Exploitation Cases - {year}.csv'
    try:
        region_data.append(process_region_data(file_path, year))
    except FileNotFoundError:
        print(f"File not found for year {year}")
        continue

# Concatenate all regional data
df_region_combined = pd.concat(region_data)

# Plot regional distribution
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_region_combined, x='Year', y='Under_12', hue='Region')
plt.title('Regional Distribution of Cases (Under 12)')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('regional_distribution_under_12.png')
plt.show()

plt.figure(figsize=(14, 8))
sns.lineplot(data=df_region_combined, x='Year', y='12_15', hue='Region')
plt.title('Regional Distribution of Cases (12-15)')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('regional_distribution_12_15.png')
plt.show()

plt.figure(figsize=(14, 8))
sns.lineplot(data=df_region_combined, x='Year', y='15_18', hue='Region')
plt.title('Regional Distribution of Cases (15-18)')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('regional_distribution_15_18.png')
plt.show()
