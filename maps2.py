# Part 1: Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
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

def create_regional_map(data, taiwan_map, year):
    """Create a focused map of Taiwan showing youth distribution with all islands"""

    # Prepare yearly data
    yearly_data = data[data['Year'] == year]
    regional_totals = yearly_data.groupby('Region')[['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']].sum().sum(axis=1)

    # Convert to projected CRS
    merged_map = taiwan_map.copy()
    merged_map = merged_map.to_crs(epsg=3824)  # TWD97 / TM2 zone 121

    # Updated name mapping dictionary including all regions
    name_mapping = {
        # Special municipalities
        'Taipei City': 'Taipei City',
        'New Taipei City': 'Taipei',
        'Taoyuan City': 'Taoyuan',
        'Taichung City': 'Taichung City',
        'Tainan City': 'Tainan City',
        'Kaohsiung City': 'Kaohsiung City',

        # Provincial cities
        'Keelung City': 'Keelung City',
        'Hsinchu City': 'Hsinchu',
        'Chiayi City': 'Chiayi',

        # Counties
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

        # Island counties
        'Penghu County': 'Penghu',
        'Kinmen County': 'Kinmen',
        'Lienchiang County': 'Lienchiang'
    }

    # Create figure with subplots for main island and smaller islands
    fig = plt.figure(figsize=(15, 12))

    # Create GridSpec for layout
    gs = plt.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

    # Main map (Taiwan island)
    ax_main = fig.add_subplot(gs[0:2, 0])

    # Inset maps for islands
    ax_penghu = fig.add_subplot(gs[0, 1])
    ax_kinmen = fig.add_subplot(gs[1, 1])

    # Map the data
    mapped_data = {}
    city_county_combined = {}

    for region, total in regional_totals.items():
        if region in name_mapping:
            shapefile_name = name_mapping[region]
            base_name = shapefile_name.replace(' City', '').replace(' County', '')

            if base_name in ['Hsinchu', 'Chiayi']:
                if base_name not in city_county_combined:
                    city_county_combined[base_name] = {'city': 0, 'county': 0, 'total': 0}
                if 'City' in region:
                    city_county_combined[base_name]['city'] = total
                else:
                    city_county_combined[base_name]['county'] = total
                city_county_combined[base_name]['total'] = (
                    city_county_combined[base_name]['city'] +
                    city_county_combined[base_name]['county']
                )
                mapped_data[base_name] = city_county_combined[base_name]['total']
            else:
                mapped_data[shapefile_name] = total

    # Add data to map
    merged_map['total_count'] = merged_map['NAME_2'].map(mapped_data)

    # Create custom colormap
    colors = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c']
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    # Plot main Taiwan island
    mainland_map = merged_map[merged_map['NAME_2'].isin([
        'Taipei City', 'Taipei', 'Taoyuan', 'Taichung City', 'Tainan City',
        'Kaohsiung City', 'Keelung City', 'Hsinchu', 'Chiayi', 'Changhwa',
        'Hualien', 'Ilan', 'Miaoli', 'Nantou', 'Pingtung', 'Taitung', 'Yunlin'
    ])]

    mainland_map.plot(
        column='total_count',
        ax=ax_main,
        legend=True,
        legend_kwds={
            'label': 'Youth Population',
            'orientation': 'horizontal',
            'shrink': 0.5,
            'aspect': 25,
            'fraction': 0.1,
            'pad': 0.05,
            'format': '{x:,.0f}'
        },
        cmap=cmap,
        missing_kwds={'color': '#f5f5f5'},
        edgecolor='white',
        linewidth=0.8
    )

    # Plot Penghu
    penghu_map = merged_map[merged_map['NAME_2'] == 'Penghu']
    penghu_map.plot(
        column='total_count',
        ax=ax_penghu,
        cmap=cmap,
        legend=False,
        edgecolor='white'
    )
    ax_penghu.set_title('Penghu County')

    # Plot Kinmen and Lienchiang
    kinmen_map = merged_map[merged_map['NAME_2'].isin(['Kinmen', 'Lienchiang'])]
    kinmen_map.plot(
        column='total_count',
        ax=ax_kinmen,
        cmap=cmap,
        legend=False,
        edgecolor='white'
    )
    ax_kinmen.set_title('Kinmen & Lienchiang Counties')

    # Add annotations to all maps
    def add_annotations(ax, data):
        for idx, row in data.iterrows():
            if pd.notnull(row['total_count']):
                point = row.geometry.representative_point()
                if row['NAME_2'] in city_county_combined:
                    label = (f"{row['NAME_2']}\n"
                            f"City: {int(city_county_combined[row['NAME_2']]['city']):,}\n"
                            f"County: {int(city_county_combined[row['NAME_2']]['county']):,}\n"
                            f"Total: {int(row['total_count']):,}")
                else:
                    label = f"{row['NAME_2']}\n{int(row['total_count']):,}"

                ax.annotate(
                    text=label,
                    xy=(point.x, point.y),
                    ha='center',
                    va='center',
                    fontsize=8,
                    bbox=dict(
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='gray',
                        boxstyle='round,pad=0.3'
                    )
                )

    # Add annotations to all maps
    add_annotations(ax_main, mainland_map)
    add_annotations(ax_penghu, penghu_map)
    add_annotations(ax_kinmen, kinmen_map)

    # Remove axes
    for ax in [ax_main, ax_penghu, ax_kinmen]:
        ax.axis('off')

    plt.suptitle(f'Taiwan Youth Population Distribution {year}', y=1.02, fontsize=16)
    plt.tight_layout()

    # Save the map
    plt.savefig(f'plots/taiwan_youth_{year}.png',
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.2)
    plt.close()

    # Print verification
    print(f"\nData Summary for {year}:")
    print(f"Total youth count: {int(regional_totals.sum()):,}")
    print(f"Total regions mapped: {sum(pd.notnull(merged_map['total_count']))}/{len(regional_totals)}")

    return merged_map

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
        data[age] = data[f'General {age}'] + data[f'Indigenous {age}']

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

    gender_colors = ['#FF69B4', '#4169E1']
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
    Create a complete map of Taiwan with main island and detailed island insets
    """
    try:
        # Step 1: Data Preparation
        yearly_data = data[data['Year'] == year]
        regional_totals = yearly_data.groupby('Region')[['Under 12 Years', '12-Under 15 Years', '15-Under 18 Years']].sum().sum(axis=1)

        # Step 2: Prepare base map
        base_map = taiwan_map.copy()
        base_map = base_map.to_crs(epsg=3824)  # TWD97 / TM2 zone 121

        # Step 3: Name mapping dictionary
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

        # Step 4: Map the data
        mapped_data = {}
        city_county_combined = {}

        for region, total in regional_totals.items():
            if region in name_mapping:
                shapefile_name = name_mapping[region]
                base_name = shapefile_name.replace(' City', '').replace(' County', '')

                if base_name in ['Hsinchu', 'Chiayi']:
                    if base_name not in city_county_combined:
                        city_county_combined[base_name] = {'City': 0, 'County': 0}
                    if 'City' in region:
                        city_county_combined[base_name]['City'] = total
                    else:
                        city_county_combined[base_name]['County'] = total
                    mapped_data[base_name] = city_county_combined[base_name]['City'] + city_county_combined[base_name]['County']
                else:
                    mapped_data[shapefile_name] = total

        # Add mapped data to base map
        base_map['total_count'] = base_map['NAME_2'].map(mapped_data)

        # Step 5: Create figure and axes
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 3, height_ratios=[1, 1, 0.2])

        # Create axes
        ax_main = fig.add_subplot(gs[:2, :2])  # Main map (2x2)
        ax_islands = fig.add_subplot(gs[:2, 2])  # Islands column

        # Turn off axes for stats
        ax_stats.axis('off')

        # Step 6: Create custom colormap
        colors = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c']
        cmap = LinearSegmentedColormap.from_list("custom", colors)

        # Step 7: Plot main Taiwan map
        main_map = base_map[base_map['NAME_2'].notna()]

        main_map.plot(
            column='total_count',
            ax=ax_main,
            legend=True,
            legend_kwds={
                'label': 'Youth Population',
                'orientation': 'horizontal',
                'shrink': 0.8,
                'aspect': 25,
                'fraction': 0.1,
                'pad': 0.05,
                'format': '{x:,.0f}'
            },
            cmap=cmap,
            missing_kwds={'color': '#f5f5f5'},
            edgecolor='white',
            linewidth=0.8
        )

        # Add annotations to main map
        for idx, row in main_map.iterrows():
            if pd.notnull(row['total_count']):
                point = row.geometry.representative_point()
                base_name = row['NAME_2'].replace(' City', '').replace(' County', '')

                if base_name in city_county_combined:
                    label = (f"{base_name}\nCity: {int(city_county_combined[base_name]['City']):,}\n"
                            f"County: {int(city_county_combined[base_name]['County']):,}\n"
                            f"Total: {int(row['total_count']):,}")
                else:
                    label = f"{row['NAME_2']}\n{int(row['total_count']):,}"

                ax_main.annotate(
                    text=label,
                    xy=(point.x, point.y),
                    ha='center',
                    va='center',
                    fontsize=8,
                    bbox=dict(
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='gray',
                        boxstyle='round,pad=0.3'
                    )
                )

        # Step 8: Plot island data
        island_regions = ['Penghu County', 'Kinmen County', 'Lienchiang County']
        island_data = []

        for region in island_regions:
            if region in regional_totals.index:
                total = regional_totals[region]
                age_data = yearly_data[yearly_data['Region'] == region].iloc[0]

                island_data.append({
                    'name': region,
                    'total': total,
                    'under_12': age_data['Under 12 Years'],
                    '12_15': age_data['12-Under 15 Years'],
                    '15_18': age_data['15-Under 18 Years']
                })

        # Create island information display
        ax_islands.axis('off')
        y_pos = 0.95

        for island in island_data:
            # Create text for island data
            island_text = (
                f"{island['name']}\n"
                f"Total: {int(island['total']):,}\n"
                f"Under 12: {int(island['under_12']):,}\n"
                f"Ages 12-15: {int(island['12_15']):,}\n"
                f"Ages 15-18: {int(island['15_18']):,}\n"
            )

            # Add text box for each island
            ax_islands.text(
                0.5, y_pos,
                island_text,
                ha='center',
                va='top',
                fontsize=10,
                bbox=dict(
                    facecolor='white',
                    edgecolor='#238b45',
                    boxstyle='round,pad=0.5'
                ),
                transform=ax_islands.transAxes
            )
            y_pos -= 0.33

        # Step 9: Add title and statistics
        plt.suptitle(f'Taiwan Youth Population Distribution {year}',
                    fontsize=20, y=0.95)

        # Add summary statistics
        total_youth = int(regional_totals.sum())
        offshore_youth = sum(regional_totals.get(region, 0) for region in island_regions)
        main_island_youth = total_youth - offshore_youth

        stats_text = (
            f"Total Youth Population: {total_youth:,}   |   "
            f"Main Island: {main_island_youth:,} ({main_island_youth/total_youth*100:.1f}%)   |   "
            f"Offshore Islands: {offshore_youth:,} ({offshore_youth/total_youth*100:.1f}%)"
        )

        ax_stats.text(0.5, 0.5, stats_text,
                     ha='center', va='center',
                     fontsize=12, fontweight='bold',
                     transform=ax_stats.transAxes)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'plots/taiwan_youth_{year}.png',
                    bbox_inches='tight',
                    dpi=300,
                    pad_inches=0.2,
                    facecolor='white')
        plt.close()

        return base_map

    except Exception as e:
        print(f"Error generating map for year {year}: {str(e)}")
        raise


def create_regional_trends(melted_data):
    """Create and save regional trends plot"""
    plt.figure(figsize=(20, 15))
    regional_yearly = melted_data.groupby(['Year', 'Region'])['Count'].sum().reset_index()

    g = sns.FacetGrid(regional_yearly, col='Region', col_wrap=4, height=4, aspect=1.5)
    g.map_dataframe(sns.lineplot, x='Year', y='Count')
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
    data = load_and_clean_data('clean_data.csv')

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
    print("Preparing Taiwan map...")
    taiwan_map = get_taiwan_shapefile()

       # Check data mapping
    # mapping_results = check_data_mapping(data, taiwan_map)
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
