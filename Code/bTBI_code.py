#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.io import mmread
import anndata as ad
from scipy.stats import fisher_exact
import math
import gseapy as gp


# **Data Formatting:**
# Read in blast and control mice scRNA seq hippocampal data and reformat into matrix where rows are cells and columns of genes (with associated UMI counts)

# In[2]:


# BLAST SAMPLES
# read in the data from file and format into correct gene expression matrix shape

# Path to the .mtx file that contains the blast UMI data
blast_umi_path = 'C:\\Users\\patso\\OneDrive\\Desktop\\20.440\\final\\blast_blast\\GSM7210821_matrix_b-TBI.mtx'

# Read in the data and store in dataframe
blast_df = pd.read_csv(blast_umi_path, skiprows=3, names=["Col 1"])

# Split the values in the "Col 1" column into three separate columns
blast_df[['Gene', 'Cell #', 'UMI']] = blast_df['Col 1'].str.split(' ', expand=True)

# Drop the original "Col 1" column
blast_df.drop(columns=['Col 1'], inplace=True)

# Path to the file that contains the gene names associated with the UMI counts
blast_gene_path = 'C:\\Users\\patso\\OneDrive\\Desktop\\20.440\\final\\blast_blast\\GSM7210821_genes_b-TBI.tsv'

# Read in the data and store in dataframe
blast_gene_df = pd.read_csv(blast_gene_path, names=["Gene #", 'Gene Name', 'etc'], sep='\t')

# Assign values to each gene name for matching in the UMI matrix
blast_gene_df['Count'] = range(1, len(blast_gene_df) + 1)

# Convert 'Count' column to object type
blast_gene_df['Count'] = blast_gene_df['Count'].astype(str)

# Merge the two DataFrames based on the 'Gene' column from the first DataFrame and the 'Count' column from the second DataFrame
merged_df = blast_df.merge(blast_gene_df[['Gene Name', 'Count']], left_on='Gene', right_on='Count', how='left')

# Create a DataFrame filled with zeros that will be filled in with the desired UMI values
zeros_df = pd.DataFrame(np.zeros((9823, len(blast_gene_df['Gene Name']))))

# Define column labels
columns = blast_gene_df['Gene Name']
# Define row labels
index = range(1, 9824)
         
# Set column and row labels
zeros_df.columns = columns
zeros_df.index = index
# Add a label to the index
zeros_df = zeros_df.rename_axis('Cell #')

# Convert 'Cell #' column to int type
merged_df['Cell #'] = merged_df['Cell #'].astype(int)

# Pivot the second matrix DataFrame to get the desired format for the gene expression matrix
pivoted_df = merged_df.pivot_table(index='Cell #', columns='Gene Name', values='UMI', fill_value=0, aggfunc='sum')

# Reindex the pivoted DataFrame to match the index of the first matrix DataFrame (if needed)
pivoted_df = pivoted_df.reindex_like(zeros_df)

# Fill missing values with 0 (if there are cells in the first matrix DataFrame not present in the second)
pivoted_df = pivoted_df.fillna(0)

# Print the pivoted DataFrame
print(pivoted_df)


# In[3]:


# CONTROL SAMPLES
# read in the data from file and format into correct gene expression matrix shape

# Path to the .mtx file that contains the control UMI data
b_ctrl_umi_path = 'C:\\Users\\patso\\OneDrive\\Desktop\\20.440\\final\\blast_ctrl\\GSM7210822_matrix_c-TBI.mtx'

# Read in the data and store in dataframe
b_ctrl_df = pd.read_csv(b_ctrl_umi_path, skiprows=3, names=["Col 1"])

# Split the values in the "Col 1" column into three separate columns
b_ctrl_df[['Gene', 'Cell #', 'UMI']] = b_ctrl_df['Col 1'].str.split(' ', expand=True)

# Drop the original "Col 1" column
b_ctrl_df.drop(columns=['Col 1'], inplace=True)

# Path to the file that contains the gene names associated with the UMI counts
b_ctrl_gene_path = 'C:\\Users\\patso\\OneDrive\\Desktop\\20.440\\final\\blast_ctrl\\GSM7210822_genes_c-TBI.tsv'

# Read in the data and store in dataframe
b_ctrl_gene_df = pd.read_csv(b_ctrl_gene_path, names=["Gene #", 'Gene Name', 'etc'], sep='\t')

# Assign values to each gene name for matching in the UMI matrix
b_ctrl_gene_df['Count'] = range(1, len(b_ctrl_gene_df) + 1)

# Convert 'Count' column to object type
b_ctrl_gene_df['Count'] = b_ctrl_gene_df['Count'].astype(str)

# Merge the two DataFrames based on the 'Gene' column from the first DataFrame and the 'Count' column from the second DataFrame
merged_df_ctrl = b_ctrl_df.merge(b_ctrl_gene_df[['Gene Name', 'Count']], left_on='Gene', right_on='Count', how='left')

# Create a DataFrame filled with zeros
zeros_df_1 = pd.DataFrame(np.zeros((7834, len(b_ctrl_gene_df['Gene Name']))))

# Define column labels
colum = b_ctrl_gene_df['Gene Name']
# Define row labels
ind = range(1, 7835)
         
# Set column and row labels
zeros_df_1.columns = colum
zeros_df_1.index = ind
# Add a label to the index
zeros_df_1 = zeros_df_1.rename_axis('Cell #')

# Convert 'Cell #' column to int type
merged_df_ctrl['Cell #'] = merged_df_ctrl['Cell #'].astype(int)

# Pivot the second matrix DataFrame to get the desired format
pivoted_df_ctrl = merged_df_ctrl.pivot_table(index='Cell #', columns='Gene Name', values='UMI', fill_value=0, aggfunc='sum')

# Reindex the pivoted DataFrame to match the index of the first matrix DataFrame (if needed)
pivoted_df_ctrl = pivoted_df_ctrl.reindex_like(zeros_df_1)

# Fill missing values with 0 (if there are cells in the first matrix DataFrame not present in the second)
pivoted_df_ctrl = pivoted_df_ctrl.fillna(0)

# Print the pivoted DataFrame
print(pivoted_df_ctrl)


# **Data Quality Control and Normalization:**
# With the read-in and formatted data tables, run quality control and normaliztion for cell cluster visualization

# In[23]:


# set parameters for scanpy figures generated
sc.settings.set_figure_params(dpi=50, facecolor="white")


# In[5]:


# Convert all values in the DataFrame to integers
pivoted_df = pivoted_df.astype(int)

# Convert all values in the DataFrame to integers
pivoted_df_ctrl = pivoted_df_ctrl.astype(int)

# empty array to combine blast and control samples
adatas = {}

# Create an AnnData object for blast object
blast_adata = sc.AnnData(pivoted_df)
blast_adata.var_names_make_unique()
adatas["blast"] = blast_adata # label those samples blast for further analysis

# Create an AnnData object for blast ctrl object
blast_ctrl_adata = sc.AnnData(pivoted_df_ctrl)
blast_ctrl_adata.var_names_make_unique()
adatas["control"] = blast_ctrl_adata # label those samples control for further analysis

# concatenate dataframes to get one large anndata onject for scanpy analysis
adata = ad.concat(adatas, label="sample")
adata.obs_names_make_unique()

# Print the AnnData object
print(adata)
print(adata.obs["sample"])


# In[6]:


# REMOVE MITOCHONDRIAL GENES AND DEAD CELLS SO ACCURATE ANALYSIS

# mitochondrial genes, "mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith("mt-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

# filter cells with less than 100 genes expressed and genes that are detected in less than 3 cells
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)

# filter for 10% mitochondrial genes
threshold_mt = 10
mito_filter = adata.obs['pct_counts_mt'] < threshold_mt
adata_blast_filtered = adata[mito_filter]

# observe qc metrics jointly - can remove cells that have too many mitochondrial genes expressed or total counts
sc.pl.violin(
    adata_blast_filtered,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)


# In[7]:


# DATA NORMALIZATION

# Saving count data
adata_blast_filtered.layers["counts"] = adata_blast_filtered.X.copy()
# Normalizing to median total counts
sc.pp.normalize_total(adata_blast_filtered)
# Logarithmize the data
sc.pp.log1p(adata_blast_filtered)


# **Analysis 1:**
# Dimensional reduction, clustering, and visulization of expression data.

# In[8]:


# reduce the dimensionality need the most informative genes - annotate highly variable genes
sc.pp.highly_variable_genes(adata_blast_filtered, n_top_genes=2000, batch_key="sample")

# reduce the dimensionality of the data by running PCA
sc.tl.pca(adata_blast_filtered, n_comps=50)

# compute neighborhood graph based on gene expression data
sc.pp.neighbors(adata_blast_filtered)

# graph can be embedded in two dimensions for UMAP visualization according to blast vs control sample type
sc.tl.umap(adata_blast_filtered)

# plot UMAP visualization in three different resolutions
for res in [0.02, 0.15, 2.0]:
    sc.tl.leiden(
        adata_blast_filtered, key_added=f"leiden_res_{res:4.2f}", resolution=res
    )



# In[9]:


# plot umap embedding of scRNA data at different resolutions and highlighted by TBI vs control sample
sc.pl.umap(
    adata_blast_filtered,
    color=["sample", "leiden_res_0.15"],
    title=["UMAP Plot Highlighted by Experimental Condition", "UMAP Plot Highlighted by Leiden Clustering"],
    save='_visuliazation_clusters.png'
)



# In[10]:


# Obtain cluster-specific differentially expressed genes
sc.tl.rank_genes_groups(adata_blast_filtered, groupby="leiden_res_0.15", method="wilcoxon")

# get dotplot highlighting top 5 DEGs in each cluster for type annotation
sc.pl.rank_genes_groups_dotplot(
    adata_blast_filtered, groupby="leiden_res_0.15", standard_scale="var", n_genes=5, 
    save='_dotplot_cell_types.png'
)


# In[11]:


# Extract top genes from clusters for cell type identification
# Define a list of group names
group_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]

# Initialize an empty dictionary to store the gene names for different groups
gene_names_dict = {}

# Iterate over each group
for group_name in group_names:
    # Extract the top 300 DEGs for the current group
    group_df = sc.get.rank_genes_groups_df(adata_blast_filtered, group=group_name).head(300)
    # Extract gene names and store them in the dictionary
    gene_names_dict[group_name] = group_df['names'].tolist()

# Access the gene names for each group using the group name as the key and set for comparison to marker set 
gene_names_0 = gene_names_dict["0"]
gene_names_1 = gene_names_dict["1"]
gene_names_2 = gene_names_dict["2"]
gene_names_3 = gene_names_dict["3"]
gene_names_4 = gene_names_dict["4"]
gene_names_5 = gene_names_dict["5"]
gene_names_6 = gene_names_dict["6"]
gene_names_7 = gene_names_dict["7"]
gene_names_8 = gene_names_dict["8"]
gene_names_9 = gene_names_dict["9"]
gene_names_10 = gene_names_dict["10"]
gene_names_11 = gene_names_dict["11"]
gene_names_12 = gene_names_dict["12"]
gene_names_13 = gene_names_dict["13"]
gene_names_14 = gene_names_dict["14"]
gene_names_15 = gene_names_dict["15"]
gene_names_16 = gene_names_dict["16"]
gene_names_17 = gene_names_dict["17"]
gene_names_18 = gene_names_dict["18"]

# Define a dictionary where each key represents a cluster and the corresponding value is a set of extracted marker genes
cluster_markers = {
    'Cluster 0': (gene_names_0),
    'Cluster 1': (gene_names_1),
    'Cluster 2': (gene_names_2),
    'Cluster 3': (gene_names_3),
    'Cluster 4': (gene_names_4),
    'Cluster 5': (gene_names_5),
    'Cluster 6': (gene_names_6), 
    'Cluster 7': (gene_names_7),
    'Cluster 8': (gene_names_8),
    'Cluster 9': (gene_names_9),
    'Cluster 10': (gene_names_10),
    'Cluster 11': (gene_names_11),
    'Cluster 12': (gene_names_12),
    'Cluster 13': (gene_names_13),
    'Cluster 14': (gene_names_14),
    'Cluster 15': (gene_names_15),
    'Cluster 16': (gene_names_16), 
    'Cluster 17': (gene_names_17),
    'Cluster 18': (gene_names_18)
}


# In[12]:


# Use top genes for cell types as defined in paper and literature stored in excel file
known_mark_path = 'C:\\Users\\patso\\OneDrive\\Desktop\\20.440\\final\\41467_2018_6222_MOESM4_ESM (5).xlsx'
df_known_markers = pd.read_excel(known_mark_path, skiprows=2)

# Extract gene names from the DataFrame column
astrocyte = df_known_markers['Astrocytes'].tolist()
endothelial = df_known_markers['Endothelial'].tolist()
ependymal = df_known_markers['Ependymal'].tolist()
microglia = df_known_markers['Microglia'].tolist()
mural = df_known_markers['Mural'].tolist()
neuron = df_known_markers['Neurons'].tolist()
oligopcs = df_known_markers['Oligodendrocyte PCs'].tolist()
oligos = df_known_markers['Oligodendrocytes'].tolist()
unknown1 = df_known_markers['Unknown1'].tolist()
unknown2 = df_known_markers['Unknown2'].tolist()

# Define a dictionary where each key represents a cell type and the corresponding value is a set of known marker genes
known_markers = {
    'Astrocytes': (astrocyte),
    'Endothelial': (endothelial),
    'Ependymal': (ependymal),
    'Microglia': (microglia),
    'Mural': (mural),
    'Neurons': (neuron),
    'Oligodendrocyte PCs': (oligopcs),
    'Oligodendrocytes': (oligos),
    'Unknown1': (unknown1),
    'Unknown2': (unknown2)
}


# In[13]:


# RUN FISHERS TEST TO ANNOTATE CELL CLUSTERS

# Dictionary to store the lowest three p-values for each cluster
lowest_p_values = {}

# Perform overlap analysis and Fisher's exact test
for cluster, markers in cluster_markers.items():
    # List to store the p-values for the current cluster
    cluster_p_values = []
    for cell_type, known_marks in known_markers.items():
        marks = set(markers)
        overlap_genes = marks.intersection(set(known_marks))
        total_genes = len(marks.union(set(known_marks)))
        overlap_count = len(overlap_genes)
        non_overlap_count = total_genes - overlap_count
        
        # Perform Fisher's exact test
        oddsratio, p_value = fisher_exact([[overlap_count, non_overlap_count],
                                            [len(marks) - overlap_count, len(known_marks) - overlap_count]])
        
        # Adjust p-value for multiple testing (Bonferroni correction)
        adjusted_p_value = p_value * len(known_marks) * len(markers)
        
        # Append the adjusted p-value to the cluster_p_values list
        cluster_p_values.append((cell_type, adjusted_p_value))
    
    # Sort the p-values for the current cluster
    cluster_p_values.sort(key=lambda x: x[1])
    
    # Store the lowest three p-values for the current cluster
    lowest_p_values[cluster] = cluster_p_values[:3]

# Print the lowest three p-values for each cluster
for cluster, p_values in lowest_p_values.items():
    print(f"Cluster {cluster} - Lowest three p-values:")
    for cell_type, p_value in p_values:
        print(f"    {cell_type}: {p_value}")


# In[14]:


# CELL TYPE ANNOTATION BASED ON FISHERS TEST AND MARKER GENES IN LITERATURE

# Define a dictionary mapping leiden clusters to cell types
cell_type_mapping = {
    0: 'Ependymal',
    1: 'Ependymal',
    2: 'Oligodendrocytes',
    3: 'Ependymal',
    4: 'Astrocytes',
    5: 'Endothelial',
    6: 'Microglia',
    7: 'Microglia',
    8: 'Oligodendrocyte PCs',
    9: 'Neurons',
    10: 'Oligodendrocytes',
    11: 'Endothelial',
    12: 'Neurons',
    13: 'Neurons',
    14: 'Neurons',
    15: 'Unknown2',
    16: 'Neurons',
    17: 'Microglia',
    18: 'Mural'
}

# Iterate over rows and update cell_type
for cell_name in adata_blast_filtered.obs.index:
    leiden_value = int(adata_blast_filtered.obs.at[cell_name, 'leiden_res_0.15'])
    if leiden_value in cell_type_mapping:
        adata_blast_filtered.obs.at[cell_name, 'cell_type'] = cell_type_mapping[leiden_value]


# In[16]:


# Determine the number of cells in each cell type 
cell_type_counts = adata_blast_filtered.obs['cell_type'].value_counts()
print(cell_type_counts)



# In[18]:


# Change name of cell type to include sample size
adata_blast_filtered.obs['cell_type'] = adata_blast_filtered.obs['cell_type'].replace({
    'Astrocytes': 'Astrocytes (N = 1227)',
    'Oligodendrocytes': 'Oligodendrocytes (N = 2547)',
    'Neurons': 'Neurons (N = 1688)',
    'Microglia': 'Microglia (N = 1679)',
    'Endothelial': 'Endothelial (N = 1482)',
    'Oligodendrocyte PCs': 'Oligodendrocyte PCs (N = 560)',
    'Ependymal': 'Ependymal (N = 8002)',
    'Mural': 'Mural (N = 51)',
    'Unknown2': 'Unknown2 (N = 246)'
})


# In[24]:


# Plot and save figures of leiden clustering by resolution, sample, and cell type annotation
sc.pl.umap(adata_blast_filtered, color='leiden_res_0.15', save='_leidan_clusters.png')  # Color UMAP plot by Leiden clusters
sc.pl.umap(adata_blast_filtered, color='sample', save='_sample_clusters.png')
sc.pl.umap(adata_blast_filtered, color='cell_type', save='_cell type_clusters.png')


# **Analysis 2:**
# DEG analysis of TBI vs control in each labelled group. Volcano plots showing the DEGs.

# In[20]:


# DEG analysis of cell types

# List of cell types for differential expression analysis of blast and control within each cell type
celltypes = ['Astrocytes (N = 1227)', 'Endothelial (N = 1482)', 'Ependymal (N = 8002)', 'Microglia (N = 1679)', 'Neurons (N = 1688)', 'Oligodendrocyte PCs (N = 560)', 
             'Oligodendrocytes (N = 2547)', 'Mural (N = 51)', 'Unknown2 (N = 246)']
colors = plt.cm.tab10.colors

# Perform differential expression analysis for each cell type
for idx,cell_type in enumerate(celltypes):
    # Make list of up regulated and down regulated hits for each cell type
    upregs = []
    downregs = []
    
    # Subset the data to include only cells of the current cell type
    adata_subset = adata_blast_filtered[adata_blast_filtered.obs['cell_type'] == cell_type]
    
    # Perform differential expression analysis
    sc.tl.rank_genes_groups(adata_subset, groupby='sample', method='wilcoxon', corr_method='benjamini-hochberg')
    
    # Retrieve DEGs for the specific group - compare blast TBI condition to control
    degs_group1 = sc.get.rank_genes_groups_df(adata_subset, group='control')
    output_file = 'degs_list_' + cell_type + '.txt'
    
    # Open the file in write mode and write the DEGs information
    with open(output_file, 'w') as file:
        # Write a header line
        file.write("Gene Names\tScores\tLogFoldChanges\tP-values\tAdjusted P-values\n")
        # Iterate through each row in degs_group1 and write gene information to the file
        for idxx, row in degs_group1.iterrows():
            file.write(f"{row['names']}\t{row['scores']}\t{row['logfoldchanges']}\t{row['pvals']}\t{row['pvals_adj']}\n")
    # Print a message indicating the file has been saved
    print(f"DEGs list has been saved to {output_file}")
    
    # Set up the volcano plot
    plt.figure(figsize=(10, 6))
    plt.scatter(degs_group1['logfoldchanges'], -1 * np.log10(degs_group1['pvals_adj']), c='gray', alpha=0.7)
    plt.title(f"Volcano Plot for {cell_type}", fontsize=20)
    plt.xlabel("log2 Fold Change", fontsize=18)
    plt.ylabel("-log10(Adjusted P-value)", fontsize=18)
    plt.axvline(x=-0.25, color='black', linestyle='--', linewidth=0.8)  # Add a vertical line at logFC=-0.25
    plt.axvline(x=0.25, color='black', linestyle='--', linewidth=0.8)  # Add a vertical line at logFC=0.25
    plt.axhline(y=-1 * np.log10(0.05), color='black', linestyle='--', linewidth=0.8)  # Add a horizontal line at max significance
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(-7.5, 7.5)
    plt.tight_layout()

    
    # Initialize lists to store top upregulated and downregulated genes
    top_upregulated_genes = []
    top_downregulated_genes = []
    
    # Iterate through the sorted DataFrame and add labels for top 10 genes in each group
    for i, row in degs_group1.iterrows():
        # filter the data to highlight genes above logfold and p-value thresholds
        if row['pvals_adj'] < 0.05 and abs(row['logfoldchanges']) > 0.25:
            if row['logfoldchanges'] > 0.25:
                top_upregulated_genes.append(row)
            else:
                top_downregulated_genes.append(row)
    
    # Sort the top upregulated and downregulated genes based on logfoldchanges
    top_upregulated_genes.sort(key=lambda x: x['logfoldchanges'], reverse=True)
    top_downregulated_genes.sort(key=lambda x: x['logfoldchanges'])
    
    # Plot all significant genes and label only the top 5 genes in each group
    for row in [top_upregulated_genes[:5], top_downregulated_genes[:5]]:
        for gene in row:
            plt.text(gene['logfoldchanges'], -1 * np.log10(gene['pvals_adj']), gene['names'], fontsize=8, ha='center', va='bottom')
    
    # Plot all genes meeting the threshold with green color
    for i, row in degs_group1.iterrows():
        if row['pvals_adj'] < 0.05 and abs(row['logfoldchanges']) > 0.25:
            if row['logfoldchanges'] > 0.25:
                plt.scatter(row['logfoldchanges'], -1 * np.log10(row['pvals_adj']), c='green', alpha=0.7)
                upregs.append(str(row['names']))
            else:
                plt.scatter(row['logfoldchanges'], -1 * np.log10(row['pvals_adj']), c='red', alpha=0.7)
                downregs.append(str(row['names']))
    
    
    # Save or show the plot
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    plt.savefig(f"volcano_plot_{cell_type}.png")  # Save the plot with a filename based on the cell type
    plt.show()  # Show the plot interactively (you can also save it instead)
    
    # Specify the file path
    file_path = cell_type + '_upregs.txt'
    # Write the list to a text file
    with open(file_path, 'w') as file:
        for item in upregs:
            file.write(f"{item}\n")
    # Specify the file path
    file_path = cell_type + '_downregs.txt'
    # Write the list to a text file
    with open(file_path, 'w') as file:
        for item in downregs:
            file.write(f"{item}\n")
    print()


# In[25]:


# 3. Visualize UMAP plot colored by the expression of a certain gene
genes_of_interest = ['Ttr', 'Ppia', 'Malat1']
sc.pl.umap(adata_blast_filtered, color=genes_of_interest, cmap='viridis', save='genesofinterest.png')


# In[93]:


# CREATE VIOLIN PLOT FOR EXPRESSION VISUALIZATION

# Create a new category combining 'cell_type' and 'sample'
adata_blast_filtered.obs['cell_type_sample'] = adata_blast_filtered.obs['cell_type'].astype(str) + '_' + adata_blast_filtered.obs['sample'].astype(str)

# Define a palette with different colors for each cell type
colors = {
    'Astrocytes (N = 1227)_control': 'darkblue',
    'Astrocytes (N = 1227)_blast': 'lightblue',
    'Neurons (N = 1688)_control': 'red',
    'Neurons (N = 1688)_blast': 'mistyrose',
    'Microglia (N = 1679)_control': 'darkgreen',
    'Microglia (N = 1679)_blast': 'lightgreen',
    'Oligodendrocytes (N = 2547)_control': 'darkorange',
    'Oligodendrocytes (N = 2547)_blast': 'bisque',
    'Endothelial (N = 1482)_control': 'indigo',
    'Endothelial (N = 1482)_blast': 'plum',
    'Oligodendrocyte PCs (N = 560)_control': 'darkcyan',
    'Oligodendrocyte PCs (N = 560)_blast': 'lightcyan',
    'Ependymal (N = 8002)_control': 'deeppink',
    'Ependymal (N = 8002)_blast': 'lightpink',
    'Mural (N = 51)_control': 'darkgoldenrod',
    'Mural (N = 51)_blast': 'lightyellow',
    'Unknown2 (N = 246)_control': 'dimgray',
    'Unknown2 (N = 246)_blast': 'lightgray'
}

# Plotting the violin plots with expression of each control vs test sample for each cell type
sc.pl.violin(adata_blast_filtered, keys='Ttr', groupby='cell_type_sample', rotation=90, stripplot=True, palette=colors, save = 'Ttr_genes_interest.png')
sc.pl.violin(adata_blast_filtered, keys='Ppia', groupby='cell_type_sample', rotation=90, stripplot=True, palette=colors, save = 'Ppia_genes_interest.png')
sc.pl.violin(adata_blast_filtered, keys='Malat1', groupby='cell_type_sample', rotation=90, stripplot=True, palette=colors, save = 'Malat1_genes_interest.png')



# **Analysis 3:** KEGG analysis of pathways of DEGs

# In[21]:


# read in the up and down DEGs of each cell type for pathway enrichment analysis
epen_up = pd.read_csv('Ependymal_upregs.txt')
epen_down = pd.read_csv('Ependymal_downregs.txt')
endo_up = pd.read_csv('Endothelial_upregs.txt')
endo_down = pd.read_csv('Endothelial_downregs.txt')
mic_up = pd.read_csv('Microglia_upregs.txt')
mic_down = pd.read_csv('Microglia_downregs.txt')
ast_up = pd.read_csv('Astrocytes_upregs.txt')
ast_down = pd.read_csv('Astrocytes_downregs.txt')
neur_up = pd.read_csv('Neurons_upregs.txt')
neur_down = pd.read_csv('Neurons_downregs.txt')
olig_up = pd.read_csv('Oligodendrocytes_upregs.txt')
olig_down = pd.read_csv('Oligodendrocytes_downregs.txt')
opc_up = pd.read_csv('Oligodendrocyte PCs_upregs.txt')
opc_down = pd.read_csv('Oligodendrocyte PCs_downregs.txt')


# In[27]:


# put in specifci gene set of cell type desired
deg_gene_list = neur_down

# Perform enrichment analysis using Enrichr for mouse genes
enrichment_results = gp.enrichr(
    gene_list=deg_gene_list,
    organism='mouse', 
    gene_sets='KEGG_2019_Mouse',  # KEGG database version for mouse
)

# Print the results
print(enrichment_results.results)


# In[ ]:





# In[ ]:





# In[ ]:




