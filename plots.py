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

def create_gender_distribution_plot(melted_data):
    """Create and save gender distribution plot"""
    plt.figure(figsize=(8.27, 5.83))  # A4 size in inches
    yearly_gender_data = melted_data.groupby(['Year', 'Gender', 'Age Group'])['Count'].sum().reset_index()

    gender_colors = {'Male': '#1f77b4', 'Female': '#ff7f0e'}
    sns.lineplot(data=yearly_gender_data, x='Year', y='Count',
                hue='Gender', style='Age Group',
                markers=True, dashes=False,
                palette=gender_colors, linewidth=2.5)

    plt.title('Gender Distribution Across Age Groups (2017-2023)', pad=15)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Demographics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/gender_distribution.png', bbox_inches='tight')
    plt.close()

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
    create_gender_distribution_plot(melted_data)

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

if __name__ == "__main__":
    main()