"""
Taiwan Youth Data Analysis and Visualization
-----------------------------------------
This script analyzes and visualizes youth population data across Taiwan's regions from 2017-2023.
Includes trend analysis, gender distribution, and geographical mapping.

Requirements:
pip install pandas numpy matplotlib seaborn geopandas requests zipfile38 shapely
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import requests
import zipfile
import io
from matplotlib.colors import LinearSegmentedColormap
import os

# Create necessary directories
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Set plotting style for publication-ready figures
def set_plotting_style():
    # plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.dpi'] = 300

# Download and prepare Taiwan shapefile
def get_taiwan_shapefile():
    """Download and prepare Taiwan shapefile from GADM"""
    shapefile_path = "data/taiwan_shapefile"

    if not os.path.exists(f"{shapefile_path}/gadm41_TWN_1.shp"):
        print("Downloading Taiwan shapefile...")
        url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_TWN_shp.zip"
        response = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(shapefile_path)
        print("Shapefile downloaded and extracted successfully.")

    return gpd.read_file(f"{shapefile_path}/gadm41_TWN_1.shp")

# Data loading and cleaning
def load_and_clean_data(file_path):
    """Load and clean the Taiwan youth data"""
    data = pd.read_csv(file_path)
    data.replace('-', np.nan, inplace=True)

    numeric_cols = [
        'General Under 12 Years', 'General 12-Under 15 Years', 'General 15-Under 18 Years',
        'Indigenous Under 12 Years', 'Indigenous 12-Under 15 Years', 'Indigenous 15-Under 18 Years'
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data[numeric_cols] = data[numeric_cols].fillna(0).astype(int)

    # Combine General and Indigenous counts
    data['Under 12 Years'] = data['General Under 12 Years'] + data['Indigenous Under 12 Years']
    data['12-Under 15 Years'] = data['General 12-Under 15 Years'] + data['Indigenous 12-Under 15 Years']
    data['15-Under 18 Years'] = data['General 15-Under 18 Years'] + data['Indigenous 15-Under 18 Years']

    return data

# Region name mapping
def create_region_mapping():
    """Create mapping between data region names and shapefile region names"""
    return {
        'New Taipei City': 'New Taipei',
        'Taipei City': 'Taipei',
        'Taoyuan City': 'Taoyuan',
        'Taichung City': 'Taichung',
        'Tainan City': 'Tainan',
        'Kaohsiung City': 'Kaohsiung',
        'Hsinchu City': 'Hsinchu City',
        'Chiayi City': 'Chiayi City',
        'Hsinchu County': 'Hsinchu',
        'Miaoli County': 'Miaoli',
        'Changhua County': 'Changhua',
        'Nantou County': 'Nantou',
        'Yunlin County': 'Yunlin',
        'Chiayi County': 'Chiayi',
        'Pingtung County': 'Pingtung',
        'Yilan County': 'Yilan',
        'Hualien County': 'Hualien',
        'Taitung County': 'Taitung',
        'Penghu County': 'Penghu',
        'Kinmen County': 'Kinmen',
        'Lienchiang County': 'Lienchiang'
    }

# Visualization functions
def plot_yearly_trends(melted_data):
    """Plot yearly trends by age group"""
    plt.figure(figsize=(12, 8))
    yearly_age_data = melted_data.groupby(['Year', 'Age Group'])['Count'].sum().reset_index()

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    sns.lineplot(data=yearly_age_data, x='Year', y='Count', hue='Age Group',
                marker='o', markersize=8, linewidth=2.5, palette=colors)

    plt.title('Trends in Youth Counts by Age Group (2017-2023)', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Total Count')
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/yearly_trend_age_groups.png', bbox_inches='tight')
    plt.close()

def plot_gender_distribution(melted_data):
    """Plot gender distribution over time"""
    plt.figure(figsize=(12, 8))
    yearly_gender_data = melted_data.groupby(['Year', 'Gender', 'Age Group'])['Count'].sum().reset_index()

    gender_colors = ['#FF69B4', '#4169E1']
    sns.lineplot(data=yearly_gender_data, x='Year', y='Count',
                hue='Gender', style='Age Group',
                markers=True, dashes=False,
                palette=gender_colors, linewidth=2.5)

    plt.title('Gender Distribution Across Age Groups (2017-2023)', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Demographics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/gender_distribution.png', bbox_inches='tight')
    plt.close()

def create_choropleth_maps(data, taiwan_map):
    """Create choropleth maps for each year"""
    for year in data['Year'].unique():
        yearly_data = data[data['Year'] == year]
        regional_totals = yearly_data.groupby('Region_mapped')[
            ['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']
        ].sum().sum(axis=1)

        merged_map = taiwan_map.merge(
            regional_totals.reset_index(),
            left_on='NAME_1',
            right_on='Region_mapped',
            how='left'
        )

        fig, ax = plt.subplots(figsize=(15, 20))

        colors = ['#f7fbff', '#08519c']
        cmap = LinearSegmentedColormap.from_list("", colors)

        merged_map.plot(
            column=0,
            ax=ax,
            legend=True,
            legend_kwds={'label': 'Total Youth Count'},
            cmap=cmap,
            missing_kwds={'color': 'lightgrey'}
        )

        for idx, row in merged_map.iterrows():
            centroid = row.geometry.centroid
            plt.annotate(
                text=row['NAME_1'],
                xy=(centroid.x, centroid.y),
                ha='center',
                fontsize=8
            )

        plt.title(f'Regional Distribution of Youth Counts ({year})', pad=20)
        plt.axis('off')
        plt.savefig(f'plots/regional_map_{year}.png', bbox_inches='tight')
        plt.close()

def create_regional_trends(melted_data):
    """Create small multiples for regional trends"""
    plt.figure(figsize=(15, 10))
    regional_yearly = melted_data.groupby(['Year', 'Region'])['Count'].sum().reset_index()

    g = sns.FacetGrid(regional_yearly, col='Region', col_wrap=4, height=3, aspect=1.5)
    g.map_dataframe(sns.lineplot, x='Year', y='Count')
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Count")

    for ax in g.axes:
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Regional Trends in Youth Counts (2017-2023)', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/regional_trends.png', bbox_inches='tight')
    plt.close()

def create_age_distribution(melted_data):
    """Create age distribution plot for latest year"""
    latest_year = melted_data['Year'].max()
    latest_data = melted_data[melted_data['Year'] == latest_year]

    plt.figure(figsize=(15, 10))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    sns.barplot(data=latest_data, x='Region', y='Count', hue='Age Group',
                palette=colors)

    plt.xticks(rotation=45, ha='right')
    plt.title(f'Age Distribution by Region ({latest_year})', pad=20)
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/age_distribution_by_region.png', bbox_inches='tight')
    plt.close()

def generate_summary_statistics(data):
    """Generate summary statistics"""
    summary_stats = data.groupby('Year')[
        ['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']
    ].agg(['mean', 'std', 'min', 'max'])

    summary_stats.to_csv('data/summary_statistics.csv')

def main():
    """Main function to run the analysis"""
    # Set up
    set_plotting_style()

    # Load and prepare data
    data = load_and_clean_data('clean_data.csv')
    taiwan_map = get_taiwan_shapefile()

    # Apply region mapping
    name_mapping = create_region_mapping()
    data['Region_mapped'] = data['Region'].map(name_mapping).fillna(data['Region'])

    # Create melted dataframe for plotting
    melted_data = data.melt(
        id_vars=['Year', 'Region', 'Region_mapped', 'Gender'],
        value_vars=['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years'],
        var_name='Age Group',
        value_name='Count'
    )

    # Generate all visualizations
    plot_yearly_trends(melted_data)
    plot_gender_distribution(melted_data)
    create_choropleth_maps(data, taiwan_map)
    create_regional_trends(melted_data)
    create_age_distribution(melted_data)
    generate_summary_statistics(data)

    print("Analysis complete! All visualizations have been saved to the 'plots' directory.")

if __name__ == "__main__":
    main()
