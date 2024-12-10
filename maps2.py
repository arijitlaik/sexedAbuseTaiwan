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
    """Create and save a regional map for a specific year with enhanced layout"""
    # Filter data for the specified year and aggregate Count per Region
    yearly_data = melted_data[melted_data['Year'] == year].copy()
    aggregated_data = yearly_data.groupby('Region')['Count'].sum().reset_index()

    # Create the region mapping
    name_mapping = create_region_mapping()

    # Map the region names in the data to the shapefile region names
    aggregated_data['MappedRegion'] = aggregated_data['Region'].map(name_mapping)

    # Drop any regions that could not be mapped
    aggregated_data.dropna(subset=['MappedRegion'], inplace=True)

    # Merge the data with the shapefile
    merged = taiwan_map.set_index('NAME_2').join(aggregated_data.set_index('MappedRegion'))

    # Check if the 'Count' column exists in the merged DataFrame
    if 'Count' not in merged.columns:
        print(f"Warning: 'Count' column not found for year {year}. Skipping map creation.")
        return

    # Initialize the figure and axes with a larger size for better visibility
    fig, ax = plt.subplots(figsize=(20, 12))

    # Define color map normalization based on data range
    norm = plt.Normalize(vmin=merged['Count'].min(), vmax=merged['Count'].max())

    # Plot the main island
    main_island = merged[merged['NAME_1'] == 'Taiwan']  # Adjust 'NAME_1' if necessary
    main_island.plot(
        column='Count',
        cmap='OrRd',
        ax=ax,
        edgecolor='black',
        linewidth=0.5,
        norm=norm,
        label='Main Island'
    )

    # Plot the smaller islands
    smaller_islands = merged[merged['NAME_1'] != 'Taiwan']
    if not smaller_islands.empty:
        smaller_islands.plot(
            column='Count',
            cmap='OrRd',
            ax=ax,
            edgecolor='black',
            linewidth=0.5,
            norm=norm,
            markersize=50,  # Adjust marker size as needed
            label='Smaller Islands'
        )

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=norm)
    sm._A = []  # Dummy array for the ScalarMappable

    # Add the colorbar to the figure
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label('Youth Count by Region')

    # Set the title and remove axis for a cleaner look
    plt.title(f"Regional Youth Counts in {year}", fontsize=20, pad=20)
    plt.axis('off')

    # Ensure the aspect ratio is equal to maintain map proportions
    ax.set_aspect('equal')

    # Calculate bounds and set limits with buffer to focus on Taiwan
    minx, miny, maxx, maxy = taiwan_map.total_bounds
    buffer_x = (maxx - minx) * 0.05
    buffer_y = (maxy - miny) * 0.05
    ax.set_xlim(minx - buffer_x, maxx + buffer_x)
    ax.set_ylim(miny - buffer_y, maxy + buffer_y)

    # Add legends for main island and smaller islands
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper left')

    # Adjust layout and save the figure
    plt.tight_layout()
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