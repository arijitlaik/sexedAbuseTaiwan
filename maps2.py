import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import requests
import tempfile
import zipfile
from matplotlib.colors import LinearSegmentedColormap
import fiona

# Create necessary directories
for directory in ['data', 'plots']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

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
    plt.figure(figsize=(15, 10))
    yearly_age_data = melted_data.groupby(['Year', 'Age Group'])['Count'].sum().reset_index()

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    sns.lineplot(data=yearly_age_data, x='Year', y='Count', hue='Age Group',
                marker='o', markersize=10, linewidth=3, palette=colors)

    plt.title('Trends in Youth Counts by Age Group (2017-2023)', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Total Count')
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/yearly_trend_age_groups.png', bbox_inches='tight')
    plt.close()

def create_gender_distribution_plot(melted_data):
    """Create and save gender distribution plot"""
    plt.figure(figsize=(15, 10))
    yearly_gender_data = melted_data.groupby(['Year', 'Gender', 'Age Group'])['Count'].sum().reset_index()

    gender_colors = {'Male': '#4169E1', 'Female': '#FF69B4'}
    sns.lineplot(data=yearly_gender_data, x='Year', y='Count',
                hue='Gender', style='Age Group',
                markers=True, dashes=False,
                palette=gender_colors, linewidth=3)

    plt.title('Gender Distribution Across Age Groups (2017-2023)', pad=20)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Demographics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/gender_distribution.png', bbox_inches='tight')
    plt.close()

def create_regional_map(melted_data, taiwan_map, year):
    """Create and save a publication-ready regional map with insets for small islands"""
    # Data preparation (previous code remains the same)
    
    # Create figure with room for insets
    fig = plt.figure(figsize=(12, 15), dpi=300)
    
    # Main map (90% of height)
    ax_main = fig.add_axes([0.1, 0.2, 0.8, 0.7])
    
    # Plot main map
    merged.plot(
        column='Count',
        cmap='OrRd',
        ax=ax_main,
        edgecolor='black',
        linewidth=0.3,
        missing_kwds={'color': 'lightgrey'},
        legend=False
    )
    
    # Add inset for Kinmen & Matsu (top right)
    ax_kinmen = fig.add_axes([0.7, 0.75, 0.25, 0.2])
    kinmen_matsu = merged[merged['NAME_1'].isin(['Kinmen', 'Lienchiang'])]
    if not kinmen_matsu.empty:
        kinmen_matsu.plot(
            column='Count',
            cmap='OrRd',
            ax=ax_kinmen,
            edgecolor='black',
            linewidth=0.3
        )
    ax_kinmen.set_title('Kinmen & Matsu', fontsize=8)
    
    # Add inset for Penghu (bottom right)
    ax_penghu = fig.add_axes([0.7, 0.15, 0.25, 0.2])
    penghu = merged[merged['NAME_1'] == 'Penghu']
    if not penghu.empty:
        penghu.plot(
            column='Count',
            cmap='OrRd',
            ax=ax_penghu,
            edgecolor='black',
            linewidth=0.3
        )
    ax_penghu.set_title('Penghu Islands', fontsize=8)
    
    # Colorbar at bottom
    sm = plt.cm.ScalarMappable(
        cmap='OrRd',
        norm=plt.Normalize(vmin=merged['Count'].min(), vmax=merged['Count'].max())
    )
    sm._A = []
    
    # Smaller colorbar
    cax = fig.add_axes([0.3, 0.1, 0.4, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Youth Count', fontsize=10, labelpad=5)
    
    # Main title
    plt.suptitle(f"Regional Youth Counts in {year}", y=0.95, fontsize=16)
    
    # Remove axes for cleaner look
    for ax in [ax_main, ax_kinmen, ax_penghu]:
        ax.axis('off')
    
    plt.savefig(f"plots/regional_map_{year}.png", bbox_inches='tight')
    plt.close()
def create_regional_trends(melted_data):
    """Create and save regional trends plot"""
    plt.figure(figsize=(20, 15))
    regional_yearly = melted_data.groupby(['Year', 'Region'])['Count'].sum().reset_index()

    g = sns.FacetGrid(regional_yearly, col='Region', col_wrap=4, height=4, aspect=1.5)
    g.map_dataframe(sns.lineplot, x='Year', y='Count', marker='o', color='#1f77b4')
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Count")

    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Regional Trends in Youth Counts (2017-2023)', y=1.02, fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/regional_trends.png', bbox_inches='tight')
    plt.close()

# Part 4: Main Execution
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

    # Optional: Analyze mapping issues
    name_mapping = create_region_mapping()
    analyze_mapping_issues(data, taiwan_map, name_mapping)

    # Create visualizations
    print("Generating visualizations...")
    create_yearly_trend_plot(melted_data)
    create_gender_distribution_plot(melted_data)

    for year in sorted(data['Year'].unique()):
        print(f"Creating map for year {year}...")
        create_regional_map(melted_data, taiwan_map, year)

    create_regional_trends(melted_data)

    # Create summary statistics
    summary_stats = data.groupby('Year')[['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']].agg(['mean', 'std', 'min', 'max'])
    summary_stats.to_csv('data/summary_statistics.csv')

    print("All visualizations have been generated and saved!")

if __name__ == "__main__":
    main()