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
    """Get Taiwan shapefile at administrative level 2 (counties and cities)"""
    cache_dir = 'data/taiwan_shapefile'
    shapefile_path = os.path.join(cache_dir, 'TWN_adm2.shp')

    if not os.path.exists(shapefile_path):
        url = "https://geodata.ucdavis.edu/diva/adm/TWN_adm.zip"
        print("Downloading Taiwan shapefile...")
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)

        os.unlink(temp_path)

    # Read the level 2 administrative shapefile
    taiwan_map = gpd.read_file(shapefile_path)

    # Set CRS
    taiwan_map.set_crs(epsg=4326, inplace=True)

    # Print information for debugging
    print("\nShapefile columns:", taiwan_map.columns.tolist())
    print("\nUnique regions in shapefile:", taiwan_map['NAME_2'].unique())

    return taiwan_map

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

def create_regional_map(data, taiwan_map, year):
    """
    Creates a map of Taiwan showing youth population distribution with improved mapping.

    Parameters:
    - data: DataFrame with youth population data by region
    - taiwan_map: GeoDataFrame with Taiwan's geographic boundaries
    - year: Year to visualize

    Returns:
    - GeoDataFrame with mapped data
    """

    # 1. Calculate youth totals by region
    yearly_data = data[data['Year'] == year]
    youth_columns = ['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']
    youth_totals = yearly_data.groupby('Region')[youth_columns].sum().sum(axis=1)

    # 2. Prepare map data
    map_data = taiwan_map.copy()
    map_data = map_data.to_crs(epsg=3824)  # Taiwan projection

    # 3. Improved region name mapping
    region_mapping = {
        # Major Cities
        'Taipei City': 'Taipei City',
        'New Taipei City': 'Taipei',
        'Taichung City': 'Taichung City',
        'Tainan City': 'Tainan City',
        'Kaohsiung City': 'Kaohsiung City',

        # Other Cities
        'Keelung City': 'Keelung City',
        # 'Hsinchu City': 'Hsinchu',
        # 'Chiayi City': 'Chiayi',
        'Taoyuan City': 'Taoyuan',

        # Counties
        'Hsinchu County': 'Hsinchu',
        'Chiayi County': 'Chiayi',
        'Changhua County': 'Changhwa',
        'Hualien County': 'Hualien',
        'Yilan County': 'Ilan',
        'Miaoli County': 'Miaoli',
        'Nantou County': 'Nantou',
        'Pingtung County': 'Pingtung',
        'Taitung County': 'Taitung',
        'Yunlin County': 'Yunlin',

        # Islands
        'Penghu County': 'Penghu',
        'Kinmen County': 'Kinmen',
        'Lienchiang County': 'Lienchiang'
    }

    # 4. Handle special cases (cities and counties with same name)
    # combined_regions = {
    #     'Hsinchu': ['Hsinchu City', 'Hsinchu County'],
    #     'Chiayi': ['Chiayi City', 'Chiayi County']
    # }

    # 5. Map the data with combined regions
    mapped_values = {}

    # Handle regular regions
    for region, total in youth_totals.items():
        if region in region_mapping:
            mapped_name = region_mapping[region]
            if mapped_name not in mapped_values:
                mapped_values[mapped_name] = total
            else:
                mapped_values[mapped_name] += total

    # 6. Add data to map
    map_data['youth_count'] = map_data['NAME_2'].map(mapped_values)

    # 7. Create visualization
    fig, ax = plt.subplots(figsize=(15, 10))

    # Create custom colormap
    colors = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b',
              '#74c476', '#41ab5d', '#238b45', '#006d2c']
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    # Plot with improved styling
    map_data.plot(
        column='youth_count',
        ax=ax,
        legend=True,
        cmap=cmap,
        legend_kwds={
            'label': 'Youth Population',
            'orientation': 'horizontal',
            'shrink': 0.7,
            'format': '{x:,.0f}'
        },
        missing_kwds={'color': '#f5f5f5'},
        edgecolor='white',
        linewidth=0.8
    )

    # Add labels for regions
    for idx, row in map_data.iterrows():
        if pd.notnull(row['youth_count']):
            point = row.geometry.representative_point()
            plt.annotate(
                f"{row['NAME_2']}\n{int(row['youth_count']):,}",
                xy=(point.x, point.y),
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none',
                    pad=0.5
                )
            )

    # Improve map appearance
    plt.title(f'Taiwan Youth Population Distribution {year}',
              fontsize=16, pad=20)
    ax.axis('off')

    # Save map
    plt.savefig(f'taiwan_Byouth_{year}.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

    return map_data

def create_regional_trends(melted_data):
    """Create and save regional trends plot"""
    plt.figure(figsize=(20, 15))
    regional_yearly = melted_data.groupby(['Year', 'Region'])['Count'].sum().reset_index()

    g = sns.FacetGrid(regional_yearly, col='Region', col_wrap=4, height=4, aspect=1.5)
    g.map_dataframe(sns.lineplot, x='Year', y='Count', marker='o')
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Count")

    for ax in g.axes:
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Regional Trends in Youth Counts (2017-2023)', y=1.02, fontsize=16)
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
    name_mapping = {
        'Taipei City': 'Taipei City',
        'New Taipei City': 'Taipei',
        'Taoyuan City': 'Taoyuan',
        'Taichung City': 'Taichung City',
        'Tainan City': 'Tainan City',
        'Kaohsiung City': 'Kaohsiung City',
        'Keelung City': 'Keelung City',
        'Hsinchu City': 'Hsinchu',
        'Chiayi City': 'Chiayi',
        'Changhua County': 'Changhwa',
        'Chiayi County': 'Chiayi',
        'Hsinchu County': 'Hsinchu',
        'Hualien County': 'Hualien',
        'Yilan County': 'Ilan',
        'Miaoli County': 'Miaoli',
        'Nantou County': 'Nantou',
        'Pingtung County': 'Pingtung',
        'Taitung County': 'Taitung',
        'Yunlin County': 'Yunlin',
        'Penghu County': 'Penghu',
        'Kinmen County': 'Kinmen',
        'Lienchiang County': 'Lienchiang'
    }
    mapping_results = analyze_mapping_issues(data, taiwan_map, name_mapping)

    # Create visualizations
    print("Generating visualizations...")
    create_yearly_trend_plot(melted_data)
    create_gender_distribution_plot(melted_data)

    for year in sorted(data['Year'].unique()):
        print(f"Creating map for year {year}...")
        create_regional_map(data, taiwan_map, year)

    create_regional_trends(melted_data)

    # Create summary statistics
    summary_stats = data.groupby('Year')[['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']].agg(['mean', 'std', 'min', 'max'])
    summary_stats.to_csv('data/summary_statistics.csv')

    print("All visualizations have been generated and saved!")

if __name__ == "__main__":
    main()
