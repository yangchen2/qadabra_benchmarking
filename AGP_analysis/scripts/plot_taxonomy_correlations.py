import pandas as pd
import biom
from biom import load_table
from biom.util import biom_open
import qiime2 as q2
import scipy.stats as ss
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging

### This script reads in the rarefied 16S and metagenomic tables and further collapses them by taxa level from gg2, then plots the taxonomy correlations

# Setup logging
logging.basicConfig(filename='../logs/plot_taxonomy_correlations.py.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Read in greengenes2 taxonomy mapping file
logging.info(f"Loading GreenGreens2 taxonomy... (this will take 1-2 minutes)")
gg_taxonomy = q2.Artifact.load('../gg_taxonomy/2022.10.taxonomy.asv.tsv.qza').view(pd.DataFrame)
logging.info(f"GreenGreens2 taxonomy loaded")

def group_by_all_taxonomy_levels(df_16S_path: str, df_metagenomic_path: str):
    """
    Read two rarefied 16S and metagenomic dataframes, attach gg2 taxonomy info, and collapse by taxonomy level to multiple BIOMs.

    Parameters:
    df_16S (pd.DataFrame): Dataframe from previously rarefied 16S table
    df_metagenomic (pd.DataFrame): Dataframe from previously rarefied metagenomic table

    Returns:
    dict: A dictionary containing 14 dataframes, 16S and metagenomic at each of 7 taxonomy levels:
    """
    logging.info("STEP 1: Converting BIOMs to dataframes.")
    # Read in BIOM tables and convert to Pandas df
    biom_table_16S = load_table(df_16S_path)
    df_16S = pd.DataFrame(biom_table_16S.to_dataframe()).transpose()
    biom_table_metagenomic = load_table(df_metagenomic_path)
    df_metagenomic = pd.DataFrame(biom_table_metagenomic.to_dataframe()).transpose()

    # Dictionary to hold the grouped dataframes for each table and each level
    grouped_dfs = {'df_16S': {}, 'df_metagenomic': {}}

    logging.info("STEP 2: Attaching taxonomy info and collapsing by taxonomy level.")
    # Process each DataFrame
    for df_name, df in zip(['df_16S', 'df_metagenomic'], [df_16S, df_metagenomic]):
        # Transpose df
        df = df.transpose()
        
        # Merge taxonomy column based on common index
        df = df.merge(gg_taxonomy['Taxon'], how='left', left_index=True, right_index=True)
        
        # Define all taxonomy levels
        taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        
        # Loop through each taxonomy level and perform grouping
        for level in taxonomy_levels:
            # Split Taxon column based on ;
            df[['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']] = df['Taxon'].str.split(';', expand=True)
            
            # Create a copy of the taxonomy_levels list and remove the current level
            levels_to_remove = taxonomy_levels.copy()
            levels_to_remove.remove(level)

            # Drop all taxonomy columns except the current level
            df_reduced = df.drop(columns=levels_to_remove)

            # Group by (collapse) features to the current taxonomy level
            df_grouped = df_reduced.groupby(level).sum()

            # Remove the ' .__' row
            # regex_pattern = r"^ .__$"
            # rows_to_drop = [index for index in df_grouped.index if re.match(regex_pattern, index)]  # Identify row indices that match the pattern
            # df_grouped = df_grouped.drop(index=rows_to_drop)

            # Store the grouped dataframe in the dictionary
            grouped_dfs[df_name][level] = df_grouped

    # Now align to common features for each level
    logging.info("STEP 3: Aligning common features at each level.")
    for level in taxonomy_levels:
        common_features = set(grouped_dfs['df_16S'][level].index).intersection(set(grouped_dfs['df_metagenomic'][level].index))
        grouped_dfs['df_16S'][level] = grouped_dfs['df_16S'][level].loc[common_features]
        grouped_dfs['df_metagenomic'][level] = grouped_dfs['df_metagenomic'][level].loc[common_features]

        # Convert and save dataframes as BIOM tables
        dataframes = {'df_16S': grouped_dfs['df_16S'][level], 'df_metagenomic': grouped_dfs['df_metagenomic'][level]}
        for name, table in dataframes.items():
            obs_ids = table.index
            samp_ids = table.columns

            biom_table = biom.table.Table(table.values, observation_ids=obs_ids, sample_ids=samp_ids)
            biom_output_file = f"../tables/{name}_collapsed_{level}.biom"

            with biom_open(biom_output_file, 'w') as f:
                biom_table.to_hdf5(f, generated_by="collapsed tables by taxonomy level and aligned features")    
    
    logging.info("-----> COMPLETED: Taxonomy collapsed BIOM files successfully converted to DataFrames")
    return grouped_dfs


def calculate_pearson_corr(grouped_dfs: dict):
    logging.info("STEP 4: Calculating Pearson correlations.")

    # Create dataframe to store correlation data
    correlation_df = pd.DataFrame()

    taxonomy_levels = ['Class', 'Order', 'Family', 'Genus', 'Species']

    # Iterate over the tables at each taxa level
    for level in taxonomy_levels:
        correlations = []
        p_values = []
        
        # Iterate over the samples
        for sample in grouped_dfs['df_16S'][level].columns:
            correlation, p_value = pearsonr(grouped_dfs['df_16S'][level][sample], grouped_dfs['df_metagenomic'][level][sample])
            correlations.append(correlation)
            p_values.append(p_value)

        correlation_df[level + '_corr'] = correlations
        correlation_df[level + '_P_value'] = p_values

    # Add Sample IDs as the first column
    correlation_df.insert(0, 'Sample', grouped_dfs['df_16S'][level].columns)

    # Drop rows with NaNs
    correlation_df = correlation_df.dropna()

    # Save correlation dataframe as tsv
    correlation_df.to_csv('../outputs/correlation_by_taxonomy.tsv', sep='\t', index=False)     
    logging.info("-----> COMPLETED: Pearson correlations saved")
    return correlation_df



def calculate_pearson_stats_plotting(correlation_df: pd.DataFrame):
    logging.info("STEP 5: Getting Pearson correlations at 25%, 50%, and 75% quartiles for plotting.")
    # Create dataframe to store correlation stats
    correlation_stats_plotting = pd.DataFrame()

    # Iterate through the correlation columns
    columns = ['Class_corr', 'Order_corr', 'Family_corr', 'Genus_corr', 'Species_corr']

    for col in columns:
        # Calculate the 25th percentile of the 'Correlation' column
        percentile_25 = correlation_df[col].quantile(0.25)
        # Calculate the mean of the 'Correlation' column
        percentile_50 = correlation_df[col].mean()
        # Calculate the 75th percentile of the 'Correlation' column
        percentile_75 = correlation_df[col].quantile(0.75)

        correlation_stats_plotting.loc["25%", col] = percentile_25
        correlation_stats_plotting.loc["50%", col] = percentile_50
        correlation_stats_plotting.loc["75%", col] = percentile_75

    logging.info("-----> COMPLETED: Pearson correlations saved")
    return correlation_stats_plotting


def plot_taxonomy_corr_line(correlation_stats_plotting: pd.DataFrame):
    logging.info("STEP 6: Plotting 16S vs metagenomic correlation by taxonomy level like line plot in Gg2 paper.")
    logging.info(correlation_stats_plotting)

    x_labels_cleaned = ['Class', 'Order', 'Family', 'Genus', 'Species']

    quartile_25 = correlation_stats_plotting.loc['25%'].tolist()
    quartile_50 = correlation_stats_plotting.loc['50%'].tolist()
    quartile_75 = correlation_stats_plotting.loc['75%'].tolist()

    blue_color = '#446faf'

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels_cleaned, quartile_25, color=blue_color, label='25%', linestyle='--', linewidth=3)
    plt.plot(x_labels_cleaned, quartile_50, color=blue_color, label='50% (Median)', linestyle='-', linewidth=3)
    plt.plot(x_labels_cleaned, quartile_75, color=blue_color, label='75%', linestyle='--', linewidth=3)

    # Axes and Labels
    plt.ylabel('Pearson Correlation', fontsize=20)
    plt.yticks([0] + [i * 0.2 for i in range(1, 6)], fontsize=18) 
    plt.ylim(0, 1.0)
    plt.xticks(range(len(x_labels_cleaned)), x_labels_cleaned, fontsize=18, va='top')
    plt.xlim(0, len(x_labels_cleaned) - 1)

    # Additional Styling
    plt.gca().spines['top']. set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.grid(False)

    # Legend
    plt.legend(fontsize=18)

    # Lower X axis text
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_y(-0.05)  # Adjust the vertical position

    # Display Plot
    plt.savefig('../figures/taxonomy_correlation_rarefaction2k9M.png') 
    logging.info("-----> COMPLETED: Plot saved!")


def plot_taxonomy_corr_box_and_violin_plot(correlation_taxonomy: str, plot_type: str, save_path: str = None):
    """
    Generates a plot (boxplot or violin plot) of Pearson correlation coefficients 
    across different taxonomic classifications from a dataset.

    This function loads a dataset from a given CSV file path, selects specific columns
    representing correlation coefficients across taxonomic classifications, and 
    dynamically generates either a boxplot or a violin plot based on the specified plot type.
    The plot is then saved to a file if a save path is provided, or displayed directly otherwise.

    Parameters:
    - correlation_taxonomy (str): Path to the CSV file containing the dataset with correlation
      coefficients. The file should be tab-separated.
    - plot_type (str): Type of plot to generate. Expected values are 'boxplot' or 'violinplot'.
      This determines whether a boxplot or violin plot is generated.
    - save_path (str, optional): Path where the generated plot should be saved. If not specified,
      the plot will be displayed directly using plt.show().

    The dataset is expected to contain the following columns: 'Class_corr', 'Order_corr', 
    'Family_corr', 'Genus_corr', 'Species_corr', representing Pearson correlation coefficients
    for different taxonomic levels.

    Example usage:
    plot_taxonomy_corr_box_and_violin_plot('path/to/data.csv', 'boxplot', 'path/to/save/plot.png')
    """
    logging.info(f"STEP 7: Plotting 16S vs metagenomic correlation by taxonomy level as {plot_type}.")

    # Load the dataset
    data = pd.read_csv(correlation_taxonomy, sep='\t')

    # Selecting only the correlation columns for plotting
    corr_columns = data[['Class_corr', 'Order_corr', 'Family_corr', 'Genus_corr', 'Species_corr']]
    x_labels_cleaned = ['Class', 'Order', 'Family', 'Genus', 'Species']

    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Create the figure for the plot
    plt.figure(figsize=(10, 6))

    # Dynamically select the plot type
    plot_func = getattr(sns, plot_type)
    my_palette = ["#4CC9F0", "#4361EE", "#3A0CA3", "#7209B7", "#F72585"]
    plot_func(data=corr_columns, showfliers=False, palette=my_palette)

    # Setting the labels and font sizes
    plt.ylabel('Pearson Correlation', size=18)
    plt.xticks(range(len(x_labels_cleaned)), x_labels_cleaned, fontsize=18, va='top')
    plt.ylim(bottom=0.0, top=1.0)

    # Add a border around the plot by adjusting the spine properties
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('black')

    # Display or save the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    logging.info(f"-----> COMPLETED: {plot_type} saved!")        



if __name__ == '__main__':
    try:
        # File paths for the BIOM files
        biom_16S_path = '../tables/df_16S_subset_rare_9k2M.biom'
        biom_metagenomic_path = '../tables/df_metagenomic_subset_rare_9k2M.biom'

        # Create collapsed taxa BIOMs
        grouped_dfs = group_by_all_taxonomy_levels(biom_16S_path, biom_metagenomic_path)

        # Create Pearson correlation dataframe 
        correlation_df = calculate_pearson_corr(grouped_dfs)

        # Create dataframe for plotting
        correlation_stats_plotting = calculate_pearson_stats_plotting(correlation_df)

        # Plot taxonomy correlation figure as line plot
        plot_taxonomy_corr_line(correlation_stats_plotting)

        # Plot taxonomy correlation figure as boxplot and violinplot
        plot_taxonomy_corr_box_and_violin_plot('../table_outputs/Correlation_by_taxonomy.tsv',
                                               'boxplot',
                                               '../figures/taxonomy_correlation_rarefaction2k9M_boxplot.png')
        plot_taxonomy_corr_box_and_violin_plot('../table_outputs/Correlation_by_taxonomy.tsv',
                                               'violinplot',
                                               '../figures/taxonomy_correlation_rarefaction2k9M_violinplot.png')



    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")