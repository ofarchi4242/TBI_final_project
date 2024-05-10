#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Import libraries for analysis
import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import core scverse libraries
import anndata as ad
import skimage


# In[2]:


df = pd.read_csv('tbi_matrix\DropSeqTBI.digital_expression.txt', sep='\t')


# In[3]:


# Transpose to get proper data structure
df_T = df.T


# In[4]:


# Adjust figure parameters for visualization
sc.settings.set_figure_params(dpi=50, facecolor="white")


# In[5]:


# Initialize scanpy data object
adata = sc.AnnData(df_T)


# In[6]:


# Loop through matrix rows (cells) and assign cells to sample IDs based on cell names
for cell_name in adata.obs.index:
    if 'Sham' in cell_name:
        adata.obs.at[cell_name, 'sample_id'] = 'Sham'
    elif 'TBI' in cell_name:
        adata.obs.at[cell_name, 'sample_id'] = 'TBI'


# In[7]:


# Assign mitochondrial genes
adata.var["mt"] = adata.var_names.str.startswith("mt-")
# Assign ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# Assign hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")


# In[8]:


# Calculate quality control metrics of adata
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)


# In[9]:


# Filter out cells that have less than 100 genes expressed and genes that are present in less than 3 cells
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)


# In[10]:


# Produce violin plots to visualize quality control metrics
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    save = '_qc.png'
)


# In[11]:


# Filter cells based on the threshold for mitochondrial gene percentage
threshold = 10  # Set your desired threshold (less than 10% mitochondrial genes)

# Create a mask for cells that meet the threshold
mito_filter = adata.obs['pct_counts_mt'] < threshold

# Filter cells in the Annotated Data object
adata_filtered = adata[mito_filter]


# In[12]:


# View violin plots again to see filtering changes
sc.pl.violin(
    adata_filtered,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)


# In[13]:


# Save count data
adata_filtered.layers["counts"] = adata_filtered.X.copy()

# Normalize to median total counts
sc.pp.normalize_total(adata_filtered)
# Logarithmize the data
sc.pp.log1p(adata_filtered)


# In[14]:


# Identify highly variable genes
sc.pp.highly_variable_genes(adata_filtered, n_top_genes=2000, batch_key="sample_id")


# In[15]:


# Plot the highly variable genes
sc.pl.highly_variable_genes(adata_filtered,save='_variable_genes.png')


# In[16]:


# Perform Principal Component Analysis (PCA)
sc.pp.pca(adata_filtered, n_comps=50, svd_solver='arpack')


# In[17]:


# Visualize variance ranking based on PCA analysis
sc.pl.pca_variance_ratio(adata_filtered, n_pcs=50, log=True, save='_variance_ratio.png')


# In[18]:


# Visualize PCs
sc.pl.pca(
    adata_filtered,
    color=["sample_id", "sample_id", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
)


# In[19]:


# Compute neighborhood relationships between cells in the dataset
sc.pp.neighbors(adata_filtered)


# In[20]:


# Compute UMAP embedding for the dataset
sc.tl.umap(adata_filtered)


# In[21]:


# Plot the UMAP dimensionality reduction
sc.pl.umap(
    adata_filtered,
    color="sample_id",
    # Setting a smaller point size to get prevent overlap
    size=2,
)


# In[22]:


# Visualize PCA results (e.g., PCA scatter plot)
sc.pl.pca(adata_filtered, color='sample_id',save='_pca.png')  # Color cells by sample ID or any other metadata


# In[23]:


# Perform Leiden clustering using the reduced-dimensional space from PCA
sc.tl.leiden(adata_filtered, resolution=0.5) 


# In[24]:


# Visualize clustering results (e.g., UMAP plot with Leiden clusters)
sc.pl.umap(adata_filtered, color='leiden')  # Color UMAP plot by Leiden clusters
sc.pl.umap(adata_filtered, color='sample_id') # Color by sample


# In[25]:


# Check different resolutions for leiden clustering
for res in [0.02, 0.5, 2.0]:
    sc.tl.leiden(
        adata_filtered, key_added=f"leiden_res_{res:4.2f}", resolution=res
    )


# In[26]:


# Visualize different leiden clustering resolutions
sc.pl.umap(
    adata_filtered,
    color=["leiden_res_0.02", "leiden_res_0.50", "leiden_res_2.00"],
    legend_loc="on data",
)


# In[27]:


# Perform differential gene expression analysis based on leiden clusters
sc.tl.rank_genes_groups(adata_filtered, groupby="leiden_res_0.50", method="wilcoxon")


# In[28]:


# Create dot plot to visualize the top five differentially expressed genes in each cluster
sc.pl.rank_genes_groups_dotplot(
    adata_filtered, groupby="leiden_res_0.50", standard_scale="var", n_genes=5, save='rank_genes_dotplot.png'
)


# In[29]:


# Define list of group names from total number of clusters
group_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9","10","11","12","13","14","15"]

# Initialize an empty dictionary to store the gene names for different groups
gene_names_dict = {}

# Iterate over each group
for group_name in group_names:
    # Extract the top 300 differentially expressed genes for the current group
    group_df = sc.get.rank_genes_groups_df(adata_filtered, group=group_name).head(200)
    # Extract gene names from the DataFrame column and store them in the dictionary
    gene_names_dict[group_name] = group_df['names'].tolist()

# Access the gene names for each group using the group name as the key
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


# In[30]:


# Import known marker genes for each cell type
df_known_markers = pd.read_excel('41467_2018_6222_MOESM4_ESM (5).xlsx', skiprows=2)


# In[31]:


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

    'Astrocytes': astrocyte, 
    'Endothelial': endothelial,
    'Ependymal': ependymal,
    'Microglia': microglia,
    'Mural': mural,
    'Neurons': neuron,
    'Oligodendrocyte PCs': oligopcs,
    'Oligodendrocytes': oligos,
    'Unknown1': unknown1,
    'Unknown2': unknown2,
    
}

# Define a dictionary where each key represents a cell type and the corresponding value is a set of known marker genes
cluster_markers = {
    'Cluster 0': gene_names_0,
    'Cluster 1': gene_names_1,
    'Cluster 2': gene_names_2,
    'Cluster 3': gene_names_3,
    'Cluster 4': gene_names_4,
    'Cluster 5': gene_names_5,
    'Cluster 6': gene_names_6, 
    'Cluster 7': gene_names_7,
    'Cluster 8': gene_names_8,
    'Cluster 9': gene_names_9,
    'Cluster 10': gene_names_10, 
    'Cluster 11': gene_names_11,
    'Cluster 12': gene_names_12,
    'Cluster 13': gene_names_13,
    'Cluster 14': gene_names_14,
    'Cluster 15': gene_names_15
}


# In[32]:


from scipy.stats import fisher_exact

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


# In[33]:


# Loop through cells and assign cell type to leiden cluster based on fisher's exact test
for cell_name in adata_filtered.obs.index:
    if 0 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Astrocytes'
    elif 1 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Oligodendrocytes'
    elif 2 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Neurons'
    elif 3 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Microglia'
    elif 4 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Endothelial'
    elif 5 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Neurons'
    elif 6 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Oligodendrocyte PCs'
    elif 7 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Ependymal'
    elif 8 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Unknown1'
    elif 9 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Oligodendrocytes'
    elif 10 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Mural'
    elif 11 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Unknown2'
    elif 12 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Microglia'
    elif 13 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Ependymal'
    elif 14 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Astrocytes'
    elif 15 == int(adata_filtered.obs.at[cell_name, 'leiden_res_0.50']):
        adata_filtered.obs.at[cell_name, 'cell_type'] = 'Neurons'


# In[34]:


# Print out total cell count in each cell type
cell_type_counts = adata_filtered.obs['cell_type'].value_counts()
print(cell_type_counts)


# In[43]:


# Change name of cell type to include sample size
adata_filtered.obs['cell_type'] = adata_filtered.obs['cell_type'].replace({
    'Astrocytes': 'Astrocytes (N = 1502)',
    'Oligodendrocytes': 'Oligodendrocytes (N = 1547)',
    'Neurons': 'Neurons (N = 1544)',
    'Microglia': 'Microglia (N = 653)',
    'Endothelial': 'Endothelial (N = 554)',
    'Oligodendrocyte PCs': 'Oligodendrocyte PCs (N = 281)',
    'Ependymal': 'Ependymal (N = 215)',
    'Unknown1': 'Unknown1 (N = 130)',
    'Mural': 'Mural (N = 97)',
    'Unknown2': 'Unknown2 (N = 78)'
})


# In[44]:


# Plot UMAPs
sc.pl.umap(adata_filtered, color='leiden_res_0.50', save='_leiden.png')  # Color UMAP plot by Leiden clusters
sc.pl.umap(adata_filtered, color='sample_id', save='_TBI_sham.png')
sc.pl.umap(adata_filtered, color='cell_type', save='_cell_type.png')


# In[48]:


from matplotlib import pyplot as plt
# Code block for generating volano plots for each cell type, TBI vs Sham


# List of cell types for DEA
celltypes = ['Astrocytes (N = 1502)', 'Endothelial (N = 554)', 'Ependymal (N = 215)', 'Microglia (N = 653)', 'Mural (N = 97)', 'Neurons (N = 1544)', 'Oligodendrocyte PCs (N = 281)', 'Oligodendrocytes (N = 1547)', 'Unknown1 (N = 130)', 'Unknown2 (N = 78)']
colors = plt.cm.tab10.colors
# Perform differential expression analysis for each cell type
for idx,cell_type in enumerate(celltypes):
    # Subset the data to include only cells of the current cell type
    adata_subset = adata_filtered[adata_filtered.obs['cell_type'] == cell_type]

    # Make list of up regulated and down regulated hits for each cell type
    upregs = []
    downregs = []
    
    # Perform differential expression analysis
    sc.tl.rank_genes_groups(adata_subset, groupby='sample_id', method='wilcoxon',adjust='fdr_bh')
    
    # Retrieve DEGs for the specific group
    degs_group1 = sc.get.rank_genes_groups_df(adata_subset, group='TBI')
    
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
    plt.xlabel("log2 Fold Change",fontsize=18)
    plt.ylabel("-log10(Adjusted P-value)",fontsize=18)
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
            
    # Add labels for significant genes
    for i, row in degs_group1.iterrows():
        if row['pvals_adj'] < 0.05 and abs(row['logfoldchanges']) > 0.25:  # Adjust significance threshold as needed
           #plt.text(row['logfoldchanges'], -1 * np.log10(row['pvals_adj']), row['names'], fontsize=8, ha='center', va='bottom')
            if row['logfoldchanges'] > 0.25:
                plt.scatter(row['logfoldchanges'], -1 * np.log10(row['pvals_adj']), c='green', alpha=0.7)
                upregs.append(str(row['names']))
            else:
                plt.scatter(row['logfoldchanges'], -1 * np.log10(row['pvals_adj']), c='red', alpha=0.7)
                downregs.append(str(row['names']))

    
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
        
    # Save or show the plot
    plt.savefig(f"volcano_plot_{cell_type}.png")  # Save the plot with a filename based on the cell type
    plt.show()  # Show the plot interactively (you can also save it instead)

    # Add a line break for better readability
    print()


# In[53]:


# Set parameters for Scanpy figures generated
sc.settings.set_figure_params(dpi=70, facecolor="white", figsize=(12, 8))  # Adjust figsize as needed
# Assuming 'adata_blast_filtered' is your AnnData object with cell type and gene expression data
# List of genes you want to plot
genes_of_interest = ['Ttr', 'Ppia', 'Malat1']
# Create a new category combining 'cell_type' and 'sample'
adata_filtered.obs['cell_type_sample'] = adata_filtered.obs['cell_type'].astype(str) + '_' + adata_filtered.obs['sample_id'].astype(str)
# Define a palette with different colors for each cell type
colors = {
    'Astrocytes (N = 1502)_Sham': 'lightblue',
    'Astrocytes (N = 1502)_TBI': 'darkblue',
    'Neurons (N = 1544)_Sham': 'mistyrose',
    'Neurons (N = 1544)_TBI': 'red',
    'Microglia (N = 653)_Sham': 'lightgreen',
    'Microglia (N = 653)_TBI': 'darkgreen',
    'Oligodendrocytes (N = 1547)_Sham': 'bisque',
    'Oligodendrocytes (N = 1547)_TBI': 'darkorange',
    'Endothelial (N = 554)_Sham': 'plum',
    'Endothelial (N = 554)_TBI': 'indigo',
    'Oligodendrocyte PCs (N = 281)_Sham': 'lightcyan',
    'Oligodendrocyte PCs (N = 281)_TBI': 'darkcyan',
    'Ependymal (N = 215)_Sham': 'lightpink',
    'Ependymal (N = 215)_TBI': 'deeppink',
    'Mural (N = 97)_Sham': 'lightyellow',
    'Mural (N = 97)_TBI': 'darkgoldenrod',
    'Unknown2 (N = 78)_Sham': 'lightgray',
    'Unknown2 (N = 78)_TBI': 'dimgray',
    'Unknown1 (N = 130)_Sham': 'white',
    'Unknown1 (N = 130)_TBI': 'black'
    # Add more cell types and corresponding colors as needed
}
# Plotting the violin plots with customized palette and custom x-axis labels
sc.pl.violin(adata_filtered, keys=genes_of_interest, groupby='cell_type_sample', rotation=90, stripplot=True, palette=colors, save='_genes_of_interest.png')


# In[55]:


# Plotting the violin plots with customized palette and custom x-axis labels
sc.pl.violin(adata_filtered, keys='Ttr', groupby='cell_type_sample', rotation=90, stripplot=True, palette=colors, save = 'Ttr_genes_interest.png')
sc.pl.violin(adata_filtered, keys='Ppia', groupby='cell_type_sample', rotation=90, stripplot=True, palette=colors, save = 'Ppia_genes_interest.png')
sc.pl.violin(adata_filtered, keys='Malat1', groupby='cell_type_sample', rotation=90, stripplot=True, palette=colors, save = 'Malat1_genes_interest.png')


# In[45]:


# Specify the file path of the text file to check for enriched pathways in each cell type (manually change for checking each)
file_path = 'Astrocytes_downregs.txt'

# Read the text file and convert its contents back to a list using list comprehension
with open(file_path, 'r') as file:
   restored_list1 = [line.strip() for line in file] # Store in variable to be checked in following block


# In[46]:


import gseapy as gp

gene_list = restored_list1

# Perform enrichment analysis using Enrichr
enrichment_results = gp.enrichr(
    gene_list=gene_list,
    organism='human',
    gene_sets='KEGG_2019_Human',  # KEGG database version
    cutoff=0.05, # Significance threshold
)


# In[47]:


# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(enrichment_results.results)


# In[48]:


# 3. Visualize UMAP plot colored by the expression of a certain gene
genes_of_interest = ['Ttr', 'Ppia', 'Malat1']
sc.pl.umap(adata_filtered, color=genes_of_interest, cmap='viridis', save='genesofinterest.png')


# In[ ]:





# In[ ]:




