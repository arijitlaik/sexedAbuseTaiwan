import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import requests
import tempfile
import zipfile
from scipy import stats

from matplotlib.colors import LinearSegmentedColormap
import fiona

# Create necessary directories
for directory in ['data', 'plots']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set style for publication-ready plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})
def analyze_gender_patterns(melted_data):
    """Detailed analysis of gender-based patterns"""


    # 1. Gender ratio trends over time
    plt.figure(figsize=(12, 6))
    gender_time = melted_data.pivot_table(
        values='Count',
        index='Year',
        columns='Gender',
        aggfunc='sum'
    )
    gender_time['Ratio (F/M)'] = gender_time['Female'] / gender_time['Male']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot absolute numbers
    gender_time[['Male', 'Female']].plot(kind='bar', ax=ax1)
    ax1.set_title('Number of Cases by Gender Over Time')
    ax1.set_ylabel('Number of Cases')

    # Plot ratio
    gender_time['Ratio (F/M)'].plot(kind='line', marker='o', ax=ax2, color='purple')
    ax2.set_title('Female to Male Ratio Over Time')
    ax2.set_ylabel('Ratio (Female/Male)')
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/gender_patterns.png')
    plt.close()

    # 2. Gender by age group analysis
    plt.figure(figsize=(10, 6))
    gender_age = pd.pivot_table(
        melted_data,
        values='Count',
        index='Age Group',
        columns='Gender',
        aggfunc='sum'
    )
    gender_age.plot(kind='bar', width=0.8)
    plt.title('Gender Distribution Across Age Groups')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/gender_age_distribution.png')
    plt.close()

    return gender_time, gender_age

def analyze_age_patterns(melted_data):
    """Detailed analysis of age-related patterns"""

    # 1. Age group trends over time with statistical analysis
    age_trends = pd.pivot_table(
        melted_data,
        values='Count',
        index='Year',
        columns='Age Group',
        aggfunc='sum'
    )

    # Calculate year-over-year changes
    yoy_changes = age_trends.pct_change() * 100

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    age_trends.plot(marker='o', ax=ax1)
    ax1.set_title('Cases by Age Group Over Time')
    ax1.set_ylabel('Number of Cases')

    yoy_changes.plot(marker='o', ax=ax2)
    ax2.set_title('Year-over-Year Change by Age Group (%)')
    ax2.set_ylabel('Percent Change')

    plt.tight_layout()
    plt.savefig('plots/age_patterns.png')
    plt.close()

    # 2. Age vulnerability index
    age_vulnerability = pd.DataFrame({
        'Total_Cases': melted_data.groupby('Age Group')['Count'].sum(),
        'Avg_Cases_per_Year': melted_data.groupby('Age Group')['Count'].mean(),
        'Max_Cases': melted_data.groupby('Age Group')['Count'].max(),
        'Growth_Rate': age_trends.iloc[-1] / age_trends.iloc[0] * 100 - 100
    }).round(2)

    return age_trends, age_vulnerability

def analyze_regional_patterns(melted_data, taiwan_map):
    """Detailed analysis of regional patterns"""

    # 1. Regional case concentration
    regional_analysis = pd.DataFrame({
        'Total_Cases': melted_data.groupby('Region')['Count'].sum(),
        'Avg_Cases_per_Year': melted_data.groupby('Region')['Count'].mean(),
        'Female_Cases': melted_data[melted_data['Gender'] == 'Female'].groupby('Region')['Count'].sum(),
        'Male_Cases': melted_data[melted_data['Gender'] == 'Male'].groupby('Region')['Count'].sum()
    })

    regional_analysis['Female_Ratio'] = (
        regional_analysis['Female_Cases'] /
        (regional_analysis['Female_Cases'] + regional_analysis['Male_Cases'])
    ).round(3)

    # 2. Regional hotspot analysis
    from scipy import stats

    regional_analysis['Z_Score'] = stats.zscore(regional_analysis['Total_Cases'])
    regional_analysis['Risk_Level'] = pd.qcut(
        regional_analysis['Z_Score'],
        q=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    # 3. Create regional heatmap
    plt.figure(figsize=(15, 10))
    regional_pivot = pd.pivot_table(
        melted_data,
        values='Count',
        index='Region',
        columns=['Year', 'Age Group'],
        aggfunc='sum',
        fill_value=0
    )

    sns.heatmap(regional_pivot,
                cmap='YlOrRd',
                annot=True,
                fmt='.0f',
                cbar_kws={'label': 'Number of Cases'})
    plt.title('Regional Case Distribution Over Time and Age Groups')
    plt.tight_layout()
    plt.savefig('plots/regional_heatmap.png')
    plt.close()

    return regional_analysis
    def create_regional_demographic_maps(melted_data, taiwan_map):
        """
        Create map heatmaps showing regional patterns by gender and age groups
        """

        # Create name mapping for regions
        name_mapping = create_region_mapping()

        def create_choropleth(data, title, filename, cmap='OrRd'):
            """Helper function to create choropleth maps"""
            data['MappedRegion'] = data['Region'].map(name_mapping)
            merged = taiwan_map.merge(
                data,
                left_on='NAME_2',
                right_on='MappedRegion',
                how='left'
            )

            fig, ax = plt.subplots(figsize=(10, 12))
            merged.plot(
                column='Count',
                cmap=cmap,
                linewidth=0.8,
                edgecolor='0.8',
                ax=ax,
                legend=True,
                legend_kwds={'label': 'Number of Cases'},
                missing_kwds={'color': 'lightgrey'}
            )

            ax.axis('off')
            ax.set_title(title, pad=20, fontsize=14)

            # Add text annotations for regions with significant numbers
            for idx, row in merged.iterrows():
                if pd.notnull(row['Count']) and row['Count'] > 0:
                    centroid = row.geometry.centroid
                    ax.annotate(
                        text=f"{int(row['Count'])}",
                        xy=(centroid.x, centroid.y),
                        horizontalalignment='center',
                        fontsize=8,
                        color='black'
                    )

            plt.savefig(f'plots/{filename}.png',
                        bbox_inches='tight',
                        dpi=300)
            plt.close()

        # 1. Create gender-based maps for each year
        years = sorted(melted_data['Year'].unique())

        for year in years:
            year_data = melted_data[melted_data['Year'] == year]

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

            # Process data for each gender
            for gender, ax in zip(['Female', 'Male'], [ax1, ax2]):
                gender_data = year_data[year_data['Gender'] == gender]
                gender_counts = gender_data.groupby('Region')['Count'].sum().reset_index()
                gender_counts['MappedRegion'] = gender_counts['Region'].map(name_mapping)

                merged = taiwan_map.merge(
                    gender_counts,
                    left_on='NAME_2',
                    right_on='MappedRegion',
                    how='left'
                )

                merged.plot(
                    column='Count',
                    cmap='OrRd',
                    linewidth=0.8,
                    edgecolor='0.8',
                    ax=ax,
                    legend=True,
                    legend_kwds={'label': 'Number of Cases'},
                    missing_kwds={'color': 'lightgrey'}
                )

                ax.axis('off')
                ax.set_title(f'{gender} Cases ({year})', pad=20, fontsize=14)

                # Add text annotations
                for idx, row in merged.iterrows():
                    if pd.notnull(row['Count']) and row['Count'] > 0:
                        centroid = row.geometry.centroid
                        ax.annotate(
                            text=f"{int(row['Count'])}",
                            xy=(centroid.x, centroid.y),
                            horizontalalignment='center',
                            fontsize=8,
                            color='black'
                        )

            plt.suptitle(f'Regional Distribution of Cases by Gender - {year}',
                         y=0.95,
                         fontsize=16)
            plt.savefig(f'plots/gender_regional_map_{year}.png',
                        bbox_inches='tight',
                        dpi=300)
            plt.close()

        # 2. Create age group maps for each year
        age_groups = melted_data['Age Group'].unique()

        for year in years:
            year_data = melted_data[melted_data['Year'] == year]

            # Calculate number of rows needed for subplots
            n_rows = (len(age_groups) + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(20, 10*n_rows))
            axes = axes.flatten()

            for idx, age_group in enumerate(age_groups):
                age_data = year_data[year_data['Age Group'] == age_group]
                age_counts = age_data.groupby('Region')['Count'].sum().reset_index()
                age_counts['MappedRegion'] = age_counts['Region'].map(name_mapping)

                merged = taiwan_map.merge(
                    age_counts,
                    left_on='NAME_2',
                    right_on='MappedRegion',
                    how='left'
                )

                merged.plot(
                    column='Count',
                    cmap='OrRd',
                    linewidth=0.8,
                    edgecolor='0.8',
                    ax=axes[idx],
                    legend=True,
                    legend_kwds={'label': 'Number of Cases'},
                    missing_kwds={'color': 'lightgrey'}
                )

                axes[idx].axis('off')
                axes[idx].set_title(f'{age_group} ({year})', pad=20, fontsize=14)

                # Add text annotations
                for _, row in merged.iterrows():
                    if pd.notnull(row['Count']) and row['Count'] > 0:
                        centroid = row.geometry.centroid
                        axes[idx].annotate(
                            text=f"{int(row['Count'])}",
                            xy=(centroid.x, centroid.y),
                            horizontalalignment='center',
                            fontsize=8,
                            color='black'
                        )

            # Remove empty subplots if any
            for idx in range(len(age_groups), len(axes)):
                fig.delaxes(axes[idx])

            plt.suptitle(f'Regional Distribution by Age Group - {year}',
                         y=0.95,
                         fontsize=16)
            plt.savefig(f'plots/{year}/age_regional_map_{year}.png',
                        bbox_inches='tight',
                        dpi=300)
            plt.close()

        # 3. Create aggregate maps for the entire period
        # Gender aggregate
        for gender in ['Female', 'Male']:
            gender_data = melted_data[melted_data['Gender'] == gender]
            total_counts = gender_data.groupby('Region')['Count'].sum().reset_index()
            create_choropleth(
                total_counts,
                f'Total {gender} Cases (2017-2023)',
                f'total_{gender.lower()}_regional_map'
            )

        # Age group aggregate
        for age_group in age_groups:
            age_data = melted_data[melted_data['Age Group'] == age_group]
            total_counts = age_data.groupby('Region')['Count'].sum().reset_index()
            create_choropleth(
                total_counts,
                f'Total Cases - {age_group} (2017-2023)',
                f'total_{age_group.replace(" ", "_").lower()}_regional_map'
            )

        return "Regional demographic maps created successfully"

def create_cross_sectional_analysis(melted_data):
    """Create cross-sectional analysis of gender, age, and region"""

    # 1. Create multi-level analysis
    cross_section = pd.pivot_table(
        melted_data,
        values='Count',
        index=['Region', 'Gender'],
        columns='Age Group',
        aggfunc='sum',
        fill_value=0
    )

    # 2. Calculate risk indices
    risk_indices = pd.DataFrame()

    # Gender disparity index
    gender_disparity = melted_data.pivot_table(
        values='Count',
        index='Region',
        columns='Gender',
        aggfunc='sum'
    )
    risk_indices['Gender_Disparity'] = (
        gender_disparity['Female'] / gender_disparity['Male']
    ).round(2)

    # Age vulnerability index
    age_totals = melted_data.pivot_table(
        values='Count',
        index='Region',
        columns='Age Group',
        aggfunc='sum'
    )
    risk_indices['Youth_Vulnerability'] = (
        age_totals['Under 12 Years'] / age_totals.sum(axis=1)
    ).round(2)

    return cross_section, risk_indices

def generate_detailed_report(gender_analysis, age_analysis, regional_analysis, risk_indices):
    """Generate a detailed statistical report"""

    report = {
        'Gender_Patterns': {
            'Overall_FM_Ratio': gender_analysis[0]['Ratio (F/M)'].mean(),
            'Trend_in_Ratio': gender_analysis[0]['Ratio (F/M)'].pct_change().mean(),
            'Most_Vulnerable_Gender': 'Female' if gender_analysis[1]['Female'].sum() > gender_analysis[1]['Male'].sum() else 'Male'
        },
        'Age_Patterns': {
            'Most_Vulnerable_Age': age_analysis[1]['Total_Cases'].idxmax(),
            'Fastest_Growing_Age_Group': age_analysis[1]['Growth_Rate'].idxmax(),
            'Average_Cases_by_Age': age_analysis[1]['Avg_Cases_per_Year'].to_dict()
        },
        'Regional_Patterns': {
            'Highest_Risk_Regions': regional_analysis[regional_analysis['Risk_Level'] == 'Very High'].index.tolist(),
            'Highest_Case_Region': regional_analysis['Total_Cases'].idxmax(),
            'Regions_Above_Average': regional_analysis[regional_analysis['Total_Cases'] > regional_analysis['Total_Cases'].mean()].index.tolist()
        }
    }

    # Save to file
    with open('data/detailed_analysis_report.txt', 'w') as f:
        f.write("Detailed Analysis of Sexual Abuse Cases in Taiwan\n")
        f.write("=" * 50 + "\n\n")

        for section, details in report.items():
            f.write(f"\n{section}\n")
            f.write("-" * len(section) + "\n")
            for key, value in details.items():
                f.write(f"{key}: {value}\n")

    return report

def analyze_reporting_patterns(melted_data):
    """Analyze temporal patterns in reporting"""
    plt.figure(figsize=(12, 6))

    # Monthly/Yearly reporting trends
    reporting_trends = melted_data.groupby(['Year', 'Region'])['Count'].sum().reset_index()

    # Create line plot with confidence intervals
    sns.lineplot(data=reporting_trends, x='Year', y='Count',
                errorbar=('ci', 95), marker='o')

    plt.title('Yearly Trend in Reported Cases')
    plt.ylabel('Number of Reports')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/reporting_trends.png')
    plt.close()

def analyze_age_vulnerability(melted_data):
    """Analyze vulnerability patterns by age group"""
    plt.figure(figsize=(10, 6))

    age_counts = melted_data.groupby('Age Group')['Count'].sum()
    total = age_counts.sum()
    percentages = (age_counts / total * 100).round(2)

    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(percentages, labels=[f'{group}\n({pct}%)' for group, pct in
            zip(percentages.index, percentages)],
            colors=colors,
            autopct='%1.1f%%')

    plt.title('Distribution of Cases by Age Group')
    plt.savefig('plots/age_vulnerability.png')
    plt.close()

def analyze_regional_risk_factors(melted_data, taiwan_map):
    """Analyze regional risk patterns and create risk index"""

    # Calculate per capita rates if population data available
    regional_totals = melted_data.groupby('Region')['Count'].sum().reset_index()

    # Calculate z-scores for identifying high-risk areas
    regional_totals['Z_Score'] = stats.zscore(regional_totals['Count'])
    regional_totals['Risk_Level'] = pd.qcut(regional_totals['Z_Score'],
                                          q=5,
                                          labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    # Save risk analysis
    regional_totals.to_csv('data/regional_risk_analysis.csv')

    return regional_totals

def analyze_intervention_points(melted_data):
    """Analyze patterns that could inform intervention strategies"""

    # Age group transition analysis
    age_progression = melted_data.pivot_table(
        values='Count',
        index='Year',
        columns='Age Group',
        aggfunc='sum'
    )

    # Plot age group transitions
    plt.figure(figsize=(12, 6))
    age_progression.plot(marker='o')
    plt.title('Case Patterns Across Age Groups Over Time')
    plt.ylabel('Number of Reports')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('plots/age_progression_analysis.png')
    plt.close()

def analyze_reporting_gaps(melted_data):
    """Analyze potential reporting gaps and disparities"""

    # Regional reporting rates analysis
    regional_stats = melted_data.groupby('Region').agg({
        'Count': ['sum', 'mean', 'std']
    }).round(2)

    # Calculate coefficient of variation to identify inconsistent reporting
    regional_stats['Count', 'cv'] = (
        regional_stats['Count', 'std'] / regional_stats['Count', 'mean']
    ).round(2)

    # Save analysis
    regional_stats.to_csv('data/reporting_consistency_analysis.csv')

    return regional_stats

def create_summary_report(melted_data, risk_analysis, reporting_analysis):
    """Create a summary report of key findings"""

    summary = {
        'Total_Cases': melted_data['Count'].sum(),
        'Years_Covered': melted_data['Year'].nunique(),
        'High_Risk_Regions': risk_analysis[
            risk_analysis['Risk_Level'].isin(['High', 'Very High'])
        ]['Region'].tolist(),
        'Age_Group_Most_Affected': melted_data.groupby('Age Group')['Count'].sum().idxmax(),
        'Gender_Ratio': melted_data[melted_data['Gender'] == 'Female']['Count'].sum() /
                       melted_data[melted_data['Gender'] == 'Male']['Count'].sum()
    }

    # Save summary
    with open('data/summary_report.txt', 'w') as f:
        f.write("Summary Report on Youth Sexual Abuse Cases\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key.replace('_', ' ')}: {value}\n")

def analyze_indigenous_cases(data):
    """Analyze patterns in indigenous communities"""

    # Calculate total cases for indigenous and general population
    indigenous_cols = [col for col in data.columns if 'Indigenous' in col]
    general_cols = [col for col in data.columns if 'General' in col]

    data['Indigenous_Total'] = data[indigenous_cols].sum(axis=1)
    data['General_Total'] = data[general_cols].sum(axis=1)

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    data.groupby('Region')[['Indigenous_Total', 'General_Total']].sum().plot(
        kind='bar',
        stacked=True
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Cases: Indigenous vs General Population')
    plt.tight_layout()
    plt.savefig('plots/indigenous_analysis.png')
    plt.close()
def get_taiwan_shapefile():
    """Get Taiwan GeoJSON at administrative level 2 (counties and cities)"""
    cache_dir = 'data/'
    geojson_zip_path = os.path.join(cache_dir, 'gadm41_TWN_2.json.zip')
    geojson_path = os.path.join(cache_dir, 'gadm41_TWN_2.json')

    if not os.path.exists(geojson_path):
        url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TWN_2.json.zip"
        print("Downloading Taiwan GeoJSON...")
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)

        os.unlink(temp_path)
        print("GeoJSON downloaded and extracted successfully.")

    # Verify the GeoJSON file exists
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found at {geojson_path}")

    # Read the GeoJSON file
    taiwan_map = gpd.read_file(geojson_path)

    # Set CRS
    taiwan_map.set_crs(epsg=4326, inplace=True)

    return taiwan_map

def create_region_mapping():
    """Create mapping between data region names and shapefile region names"""
    return {
        'Changhua County': 'Changhua',
        'Chiayi City': 'ChiayiCity',
        'Chiayi County': 'ChiayiCounty',
        'Hsinchu City': 'HsinchuCity',
        'Hsinchu County': 'HsinchuCounty',
        'Hualien County': 'Hualien',
        'Kaohsiung City': 'Kaohsiung',
        'Keelung City': 'Keelung',
        'Kinmen County': 'Kinmen',
        'Lienchiang County': 'Lienkiang',
        'Miaoli County': 'Miaoli',
        'Nantou County': 'Nantou',
        'New Taipei City': 'NewTaipei',
        'Penghu County': 'Penghu',
        'Pingtung County': 'Pingtung',
        'Taichung City': 'Taichung',
        'Tainan City': 'Tainan',
        'Taipei City': 'Taipei',
        'Taitung County': 'Taitung',
        'Taoyuan City': 'Taoyuan',
        'Yilan County': 'Yilan',
        'Yunlin County': 'Yulin'
    }

def analyze_mapping_issues(data, taiwan_map, name_mapping):
    """Analyze which regions are not being mapped and why"""
    print("\nMapping Analysis:")
    print("-" * 50)

    # Get all unique regions from data
    data_regions = set(data['Region'].unique())

    # Get all unique regions from shapefile
    shapefile_regions = set(taiwan_map['NAME_2'].dropna().unique())

    print("\n1. Regions in your data:")
    print(sorted(data_regions))

    print("\n2. Regions in shapefile:")
    print(sorted(shapefile_regions))

    print("\n3. Regions in mapping dictionary:")
    print(sorted(name_mapping.keys()))

    # Find unmapped regions
    unmapped_regions = data_regions - set(name_mapping.keys())
    print("\n4. Regions in data that are not in mapping dictionary:")
    print(sorted(unmapped_regions))

    # Find mapped regions that don't match shapefile
    incorrect_mappings = set()
    for data_region, shapefile_region in name_mapping.items():
        if shapefile_region not in shapefile_regions:
            incorrect_mappings.add((data_region, shapefile_region))

    print("\n5. Incorrect mappings (mapped to non-existent shapefile regions):")
    for data_region, shapefile_region in sorted(incorrect_mappings):
        print(f"'{data_region}' -> '{shapefile_region}'")

    return {
        'unmapped_regions': unmapped_regions,
        'incorrect_mappings': incorrect_mappings
    }

def load_and_clean_data(file_path):
    """Load and clean the dataset"""
    data = pd.read_csv(file_path)
    data.replace('-', np.nan, inplace=True)

    numeric_cols = [col for col in data.columns if 'Years' in col]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data[numeric_cols] = data[numeric_cols].fillna(0).astype(int)

    # Combine General and Indigenous counts
    age_groups = ['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']
    for age in age_groups:
        general_col = f'General {age}'
        indigenous_col = f'Indigenous {age}'
        if general_col in data.columns and indigenous_col in data.columns:
            data[age] = data[general_col] + data[indigenous_col]
        else:
            print(f"Warning: Columns {general_col} or {indigenous_col} not found in data.")
            data[age] = data.get(general_col, 0) + data.get(indigenous_col, 0)

    return data

# Part 3: Visualization Functions
def create_yearly_trend_plot(melted_data):
    """Create and save yearly trend plot"""
    plt.figure(figsize=(8.27, 5.83))  # A4 size in inches (width, height)
    yearly_age_data = melted_data.groupby(['Year', 'Age Group'])['Count'].sum().reset_index()

    sns.lineplot(data=yearly_age_data, x='Year', y='Count', hue='Age Group',
                marker='o', markersize=6, linewidth=2.5)

    plt.title('Trends in Youth Counts by Age Group (2017-2023)', pad=15)
    plt.xlabel('Year')
    plt.ylabel('Total Count')
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/yearly_trend_age_groups.png', bbox_inches='tight')
    plt.close()

# def create_gender_distribution_plot(melted_data):
#     """Create and save gender distribution plot"""
#     plt.figure(figsize=(8.27, 5.83))  # A4 size in inches
#     yearly_gender_data = melted_data.groupby(['Year', 'Gender', 'Age Group'])['Count'].sum().reset_index()

#     gender_colors = {'Male': '#1f77b4', 'Female': '#ff7f0e'}
#     sns.lineplot(data=yearly_gender_data, x='Year', y='Count',
#                 hue='Gender', style='Age Group',
#                 markers=True, dashes=False,
#                 palette=gender_colors, linewidth=2.5)

#     plt.title('Gender Distribution Across Age Groups (2017-2023)', pad=15)
#     plt.xlabel('Year')
#     plt.ylabel('Count')
#     plt.legend(title='Demographics', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.savefig('plots/gender_distribution.png', bbox_inches='tight')
#     plt.close()

def create_regional_map(melted_data, taiwan_map, year):
    """Create and save a publication-ready regional map for a specific year"""
    # Filter and prepare data
    yearly_data = melted_data[melted_data['Year'] == year].groupby('Region')['Count'].sum().reset_index()
    name_mapping = create_region_mapping()
    yearly_data['MappedRegion'] = yearly_data['Region'].map(name_mapping)
    yearly_data.dropna(subset=['MappedRegion'], inplace=True)

    # Merge data with shapefile
    merged = taiwan_map.set_index('NAME_2').join(yearly_data.set_index('MappedRegion'))

    if 'Count' not in merged.columns:
        print(f"Warning: No data for year {year}")
        return

    # Define figure size for A4
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)  # A4 size

    # Plot the map with enhanced aesthetics
    merged.plot(
        column='Count',
        cmap='OrRd',
        ax=ax,
        edgecolor='black',
        linewidth=0.5,
        missing_kwds={'color': 'lightgrey'}
    )

    # Add a smaller, horizontally oriented colorbar
    sm = plt.cm.ScalarMappable(
        cmap='OrRd',
        norm=plt.Normalize(vmin=merged['Count'].min(), vmax=merged['Count'].max())
    )
    sm._A = []

    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.025, pad=0.05)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Youth Count', fontsize=12, labelpad=3)

    # Set title
    plt.title(f"Regional Youth Counts in {year}", fontsize=16, pad=20)

    # Remove axes for cleaner look
    ax.axis('off')

    # Adjust layout
    minx, miny, maxx, maxy = taiwan_map.total_bounds

    buffer_x = (maxx - minx) * 0.001
    buffer_y = (maxy - miny) * 0.001
    ax.set_xlim(118 - buffer_x, maxx + buffer_x)
    ax.set_ylim(miny+1 - buffer_y, maxy + buffer_y)
    # Save the figure
    plt.savefig(f"plots/regional_map_{year}.png", bbox_inches='tight')
    plt.close()

def create_gender_variation_map(melted_data, taiwan_map, year):
    """Create and save a publication-ready gender variation map for a specific year or aggregate"""
    # Aggregate data by Region and Gender
    if year == 'aggregate':
        aggregated_data = melted_data.groupby(['Region', 'Gender'])['Count'].sum().reset_index()
        title_suffix = 'Aggregate (2017-2023)'
    else:
        aggregated_data = melted_data[melted_data['Year'] == year].groupby(['Region', 'Gender'])['Count'].sum().reset_index()
        title_suffix = f"{year}"

    # Create the region mapping
    name_mapping = create_region_mapping()
    aggregated_data['MappedRegion'] = aggregated_data['Region'].map(name_mapping)
    aggregated_data.dropna(subset=['MappedRegion'], inplace=True)

    # Merge data with shapefile
    merged = taiwan_map.merge(
        aggregated_data,
        left_on='NAME_2',
        right_on='MappedRegion',
        how='left'
    )

    # Handle missing data
    merged['Count'] = merged['Count'].fillna(0)

    # Initialize the figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=300)  # Landscape A4
    cmap = 'OrRd'
    norm = plt.Normalize(vmin=merged['Count'].min(), vmax=merged['Count'].max())

    # Plot Male map
    male_data = merged[merged['Gender'] == 'Male']
    male_data.plot(
        column='Count',
        cmap=cmap,
        linewidth=0.5,
        edgecolor='black',
        ax=axes[0],
        norm=norm,
        missing_kwds={'color': 'lightgrey'}
    )
    axes[0].set_title('Male Youth Count', fontsize=14)
    axes[0].axis('off')

    # Plot Female map
    female_data = merged[merged['Gender'] == 'Female']
    female_data.plot(
        column='Count',
        cmap=cmap,
        linewidth=0.5,
        edgecolor='black',
        ax=axes[1],
        norm=norm,
        missing_kwds={'color': 'lightgrey'}
    )
    axes[1].set_title('Female Youth Count', fontsize=14)
    axes[1].axis('off')

    # Add a unified colorbar below the subplots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Youth Count', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Set the main title
    plt.suptitle(f"Gender Variation in Youth Counts - {title_suffix}", fontsize=16, y=0.95)

    # Adjust layout for better spacing
    minx, miny, maxx, maxy = taiwan_map.total_bounds
    print(minx, miny, maxx, maxy)
    buffer_x = (maxx - minx) * 0.001
    buffer_y = (maxy - miny) * 0.001

    for ax in axes:
        ax.set_xlim(118 - buffer_x, maxx + buffer_x)
        ax.set_ylim(miny+1 - buffer_y, maxy + buffer_y)

    # Save the figure
    if year == 'aggregate':
        plt.savefig(f"plots/gender_variation_map_aggregate.png", bbox_inches='tight')
    else:
        plt.savefig(f"plots/gender_variation_map_{year}.png", bbox_inches='tight')
    plt.close()

def create_regional_trends(melted_data):
    """Create and save regional trends plot"""
    plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
    regional_yearly = melted_data.groupby(['Year', 'Region'])['Count'].sum().reset_index()

    g = sns.FacetGrid(regional_yearly, col='Region', col_wrap=4, height=2.5, aspect=1.2)
    g.map_dataframe(sns.lineplot, x='Year', y='Count', marker='o', color='#1f77b4')
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Count")

    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('Regional Trends in Youth Counts (2017-2023)', y=1.02, fontsize=18)
    plt.savefig('plots/regional_trends.png', bbox_inches='tight')
    plt.close()


def create_regional_demographic_maps(melted_data, taiwan_map):
    """
    Create map heatmaps showing regional patterns by gender and age groups
    with consistent styling and layout
    """
    # Create name mapping for regions
    name_mapping = create_region_mapping()

    # Get global min and max for consistent color scaling
    vmin = melted_data['Count'].min()
    vmax = melted_data['Count'].max()

    # Get map bounds
    minx, miny, maxx, maxy = taiwan_map.total_bounds
    buffer_x = (maxx - minx) * 0.001
    buffer_y = (maxy - miny) * 0.001

    def adjust_map_bounds(ax):
        """Helper function to set consistent map bounds"""
        ax.set_xlim(118 - buffer_x, maxx + buffer_x)
        ax.set_ylim(miny+1 - buffer_y, maxy + buffer_y)

    def create_choropleth(data, title, filename):
        """Helper function to create choropleth maps"""
        data['MappedRegion'] = data['Region'].map(name_mapping)
        merged = taiwan_map.merge(
            data,
            left_on='NAME_2',
            right_on='MappedRegion',
            how='left'
        )

        # Define figure size for A4
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)

        merged.plot(
            column='Count',
            cmap='OrRd',
            linewidth=0.5,
            edgecolor='black',
            ax=ax,
            missing_kwds={'color': 'lightgrey'},
            vmin=vmin,
            vmax=vmax
        )

        # Add a smaller, horizontally oriented colorbar
        sm = plt.cm.ScalarMappable(
            cmap='OrRd',
            norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.025, pad=0.05)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Number of Cases', fontsize=12, labelpad=3)

        ax.axis('off')
        ax.set_title(title, pad=20, fontsize=14)

        # Add text annotations
        for idx, row in merged.iterrows():
            if pd.notnull(row['Count']) and row['Count'] > 0:
                centroid = row.geometry.centroid
                ax.annotate(
                    text=f"{int(row['Count'])}",
                    xy=(centroid.x, centroid.y),
                    horizontalalignment='center',
                    fontsize=8,
                    color='black'
                )

        adjust_map_bounds(ax)
        plt.savefig(f'plots/{filename}.png', bbox_inches='tight')
        plt.close()

    # 1. Create gender-based maps for each year
    years = sorted(melted_data['Year'].unique())

    for year in years:
        year_data = melted_data[melted_data['Year'] == year]

        # Create figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=300)  # Landscape A4

        # Create unified norm for consistent color scaling
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Process data for each gender
        for gender, ax in zip(['Female', 'Male'], axes):
            gender_data = year_data[year_data['Gender'] == gender]
            gender_counts = gender_data.groupby('Region')['Count'].sum().reset_index()
            gender_counts['MappedRegion'] = gender_counts['Region'].map(name_mapping)

            merged = taiwan_map.merge(
                gender_counts,
                left_on='NAME_2',
                right_on='MappedRegion',
                how='left'
            )

            merged['Count'] = merged['Count'].fillna(0)

            merged.plot(
                column='Count',
                cmap='OrRd',
                linewidth=0.5,
                edgecolor='black',
                ax=ax,
                norm=norm,
                missing_kwds={'color': 'lightgrey'}
            )

            ax.axis('off')
            ax.set_title(f'{gender} Cases ({year})', pad=20, fontsize=14)
            adjust_map_bounds(ax)

            # Add text annotations
            for idx, row in merged.iterrows():
                if pd.notnull(row['Count']) and row['Count'] > 0:
                    centroid = row.geometry.centroid
                    ax.annotate(
                        text=f"{int(row['Count'])}",
                        xy=(centroid.x, centroid.y),
                        horizontalalignment='center',
                        fontsize=8,
                        color='black'
                    )

        # Add unified colorbar
        sm = plt.cm.ScalarMappable(cmap='OrRd', norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label('Number of Cases', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        plt.suptitle(f'Regional Distribution of Cases by Gender - {year}',
                    fontsize=16, y=0.95)
        plt.savefig(f'plots/gender_regional_map_{year}.png', bbox_inches='tight')
        plt.close()

    # 2. Create age group maps for each year
    age_groups = sorted(melted_data['Age Group'].unique())

    for year in years:
        year_data = melted_data[melted_data['Year'] == year]

        # Calculate number of rows needed for subplots (3 age groups = 2 rows, 2 columns)
        fig, axes = plt.subplots(1, 3, figsize=(16, 8), dpi=300)  # Landscape layout with 3 columns
        axes = axes.flatten()

        # Create unified norm for consistent color scaling
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        for idx, age_group in enumerate(age_groups):
            age_data = year_data[year_data['Age Group'] == age_group]
            age_counts = age_data.groupby('Region')['Count'].sum().reset_index()
            age_counts['MappedRegion'] = age_counts['Region'].map(name_mapping)

            merged = taiwan_map.merge(
                age_counts,
                left_on='NAME_2',
                right_on='MappedRegion',
                how='left'
            )

            merged['Count'] = merged['Count'].fillna(0)

            merged.plot(
                column='Count',
                cmap='OrRd',
                linewidth=0.5,
                edgecolor='black',
                ax=axes[idx],
                norm=norm,
                missing_kwds={'color': 'lightgrey'}
            )

            axes[idx].axis('off')
            axes[idx].set_title(f'{age_group} ({year})', pad=20, fontsize=14)
            adjust_map_bounds(axes[idx])

            # Add text annotations
            for _, row in merged.iterrows():
                if pd.notnull(row['Count']) and row['Count'] > 0:
                    centroid = row.geometry.centroid
                    axes[idx].annotate(
                        text=f"{int(row['Count'])}",
                        xy=(centroid.x, centroid.y),
                        horizontalalignment='center',
                        fontsize=8,
                        color='black'
                    )

        # Remove extra subplot if any
        if len(age_groups) < len(axes):
            fig.delaxes(axes[-1])

        # Add unified colorbar
        sm = plt.cm.ScalarMappable(cmap='OrRd', norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label('Number of Cases', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        plt.suptitle(f'Regional Distribution by Age Group - {year}',
                    fontsize=16, y=0.95)
        plt.savefig(f'plots/age_regional_map_{year}.png', bbox_inches='tight')
        plt.close()

    # 3. Create aggregate maps for the entire period
    # Gender aggregate (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=300)  # Landscape A4
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for idx, gender in enumerate(['Female', 'Male']):
        gender_data = melted_data[melted_data['Gender'] == gender]
        total_counts = gender_data.groupby('Region')['Count'].sum().reset_index()
        total_counts['MappedRegion'] = total_counts['Region'].map(name_mapping)

        merged = taiwan_map.merge(
            total_counts,
            left_on='NAME_2',
            right_on='MappedRegion',
            how='left'
        )

        merged['Count'] = merged['Count'].fillna(0)

        merged.plot(
            column='Count',
            cmap='OrRd',
            linewidth=0.5,
            edgecolor='black',
            ax=axes[idx],
            norm=norm,
            missing_kwds={'color': 'lightgrey'}
        )

        axes[idx].axis('off')
        axes[idx].set_title(f'Total {gender} Cases (2017-2023)',
                          pad=20, fontsize=14)
        adjust_map_bounds(axes[idx])

        # Add text annotations
        for _, row in merged.iterrows():
            if pd.notnull(row['Count']) and row['Count'] > 0:
                centroid = row.geometry.centroid
                axes[idx].annotate(
                    text=f"{int(row['Count'])}",
                    xy=(centroid.x, centroid.y),
                    horizontalalignment='center',
                    fontsize=8,
                    color='black'
                )

    # Add unified colorbar
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Total Number of Cases', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle('Aggregate Regional Distribution by Gender (2017-2023)',
                fontsize=16, y=0.95)
    plt.savefig('plots/total_gender_regional_map.png', bbox_inches='tight')
    plt.close()

    # Age group aggregate maps
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), dpi=300)  # Landscape layout with 3 columns
    axes = axes.flatten()

    for idx, age_group in enumerate(age_groups):
        age_data = melted_data[melted_data['Age Group'] == age_group]
        total_counts = age_data.groupby('Region')['Count'].sum().reset_index()
        total_counts['MappedRegion'] = total_counts['Region'].map(name_mapping)

        merged = taiwan_map.merge(
            total_counts,
            left_on='NAME_2',
            right_on='MappedRegion',
            how='left'
        )

        merged['Count'] = merged['Count'].fillna(0)

        merged.plot(
            column='Count',
            cmap='OrRd',
            linewidth=0.5,
            edgecolor='black',
            ax=axes[idx],
            norm=norm,
            missing_kwds={'color': 'lightgrey'}
        )

        axes[idx].axis('off')
        axes[idx].set_title(f'Total CASES - {age_group} (2017-2023)',
                          pad=20, fontsize=14)

        adjust_map_bounds(axes[idx])

        # Add text annotations
        for _, row in merged.iterrows():
            if pd.notnull(row['Count']) and row['Count'] > 0:
                centroid = row.geometry.centroid
                axes[idx].annotate(
                    text=f"{int(row['Count'])}",
                    xy=(centroid.x, centroid.y),
                    horizontalalignment='center',
                    fontsize=8,
                    color='black'
                )

    # Remove extra subplot if any
    if len(age_groups) < len(axes):
        fig.delaxes(axes[-1])

    # Add unified colorbar for age group aggregate maps
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Total Number of Cases', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle('Aggregate Regional Distribution by Age Group (2017-2023)',
                fontsize=16, y=0.95)
    plt.savefig('plots/total_age_regional_map.png', bbox_inches='tight')
    plt.close()

    # 4. Create a combined statistics summary
    summary_stats = {
        'Total Cases': melted_data['Count'].sum(),
        'Years Covered': len(years),
        'Gender Ratio (F/M)': (
            melted_data[melted_data['Gender'] == 'Female']['Count'].sum() /
            melted_data[melted_data['Gender'] == 'Male']['Count'].sum()
        ).round(2),
        'Most Affected Region': melted_data.groupby('Region')['Count'].sum().idxmax(),
        'Most Affected Age Group': melted_data.groupby('Age Group')['Count'].sum().idxmax()
    }

    # Save summary statistics
    with open('data/mapping_summary_stats.txt', 'w') as f:
        f.write("Regional Mapping Analysis Summary\n")
        f.write("================================\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    print("Regional demographic maps and analysis completed successfully")
    return summary_stats

def main():
    # Load and prepare data
    print("Loading and cleaning data...")
    data = load_and_clean_data('data/clean_data.csv')  # Ensure the correct path

    # Prepare melted data for plotting
    melted_data = data.melt(
        id_vars=['Year', 'Region', 'Gender'],
        value_vars=['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years'],
        var_name='Age Group',
        value_name='Count'
    )

    # Get Taiwan shapefile
    print("Preparing Taiwan map...")
    taiwan_map = get_taiwan_shapefile()

    # Analyze mapping issues
    name_mapping = create_region_mapping()
    analyze_mapping_issues(data, taiwan_map, name_mapping)

    # Create visualizations
    print("Generating visualizations...")
    create_yearly_trend_plot(melted_data)
    # create_gender_distribution_plot(melted_data)

    # Generate regional maps for each year
    for year in sorted(data['Year'].unique()):
        print(f"Creating regional map for year {year}...")
        create_regional_map(melted_data, taiwan_map, year)
        print(f"Creating gender variation map for year {year}...")
        create_gender_variation_map(melted_data, taiwan_map, year)

    # Generate aggregate gender variation map
    print("Creating aggregate gender variation map...")
    create_gender_variation_map(melted_data, taiwan_map, 'aggregate')

    # Create regional trends
    create_regional_trends(melted_data)

    # Create summary statistics
    summary_stats = data.groupby('Year')[['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']].agg(['mean', 'std', 'min', 'max'])
    summary_stats.to_csv('data/summary_statistics.csv')

    print("All visualizations have been generated and saved!")
    # Perform specialized analyses
    risk_analysis = analyze_regional_risk_factors(melted_data, taiwan_map)
    reporting_analysis = analyze_reporting_gaps(melted_data)

    # Create visualizations
    analyze_reporting_patterns(melted_data)
    analyze_age_vulnerability(melted_data)
    analyze_intervention_points(melted_data)
    analyze_indigenous_cases(data)

    # Generate summary report
    create_summary_report(melted_data, risk_analysis, reporting_analysis)
    print("Creating regional demographic maps...")
    create_regional_demographic_maps(melted_data, taiwan_map)
    print("Analysis completed. Please check the data folder for detailed reports.")
    # Perform detailed analyses
    gender_analysis = analyze_gender_patterns(melted_data)
    age_analysis = analyze_age_patterns(melted_data)
    regional_analysis = analyze_regional_patterns(melted_data, taiwan_map)
    cross_section, risk_indices = create_cross_sectional_analysis(melted_data)

    # Generate detailed report
    report = generate_detailed_report(
        gender_analysis,
        age_analysis,
        regional_analysis,
        risk_indices
    )

    # Save additional data
    cross_section.to_csv('data/cross_sectional_analysis.csv')
    risk_indices.to_csv('data/risk_indices.csv')

    print("Detailed analysis completed. Check the data and plots folders for results.")

if __name__ == "__main__":
    main()
