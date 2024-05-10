#!/usr/bin/env python
# coding: utf-8

# In[112]:


import scanpy
import numpy as np
import pandas as pd
import os


# In[113]:


# Define the directory containing the CSV files
fp_degs_directory = '/'

# Initialize a list to store the data from CSV files
fp_data_upregs = []

# Loop through each CSV file in the directory
for filename in os.listdir():
    if filename.endswith('upregs.txt'):
        try:
            df = pd.read_csv(filename,header=None)
            # Check if the DataFrame is not empty
            if not df.empty:
                # Convert each column of the DataFrame to a list and append to csv_data
                fp_data_upregs.append(df.values.tolist())
                print(filename)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {filename}")
        



# In[114]:


fp_data_upregs


# In[115]:


# Initialize a list to store the data from CSV files
fp_data_downregs = []

# Loop through each CSV file in the directory
for filename in os.listdir():
    if filename.endswith('downregs.txt'):
        try:
            df = pd.read_csv(filename,header=None)
            # Check if the DataFrame is not empty
            if not df.empty:
                # Convert each column of the DataFrame to a list and append to csv_data
                fp_data_downregs.append(df.values.tolist())
                print(filename)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {filename}")
        


# In[116]:


fp_data_downregs


# In[117]:


# Initialize a list to store the data from CSV files
b_data_upregs = []

# Loop through each CSV file in the directory
for filename in os.listdir('blast_degs_1/'):
    if filename.endswith('upregs.txt'):
        try:
            df = pd.read_csv(os.path.join('blast_degs_1/', filename), header=None)
            # Check if the DataFrame is not empty
            if not df.empty:
                # Convert each column of the DataFrame to a list and append to csv_data
                b_data_upregs.append(df.values.tolist())
                print(filename)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {filename}")
        


# In[118]:


b_data_upregs


# In[119]:


# Initialize a list to store the data from CSV files
b_data_downregs = []

# Loop through each CSV file in the directory
for filename in os.listdir('blast_degs_1/'):
    if filename.endswith('downregs.txt'):
        try:
            df = pd.read_csv(os.path.join('blast_degs_1/', filename), header=None)
            # Check if the DataFrame is not empty
            if not df.empty:
                # Convert each column of the DataFrame to a list and append to csv_data
                b_data_downregs.append(df.values.tolist())
                print(filename)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {filename}")
        


# In[120]:


b_data_downregs


# In[121]:


for list_num in range(7):
    intersection = list(set(map(tuple, fp_data_upregs[list_num])) & set(map(tuple, b_data_upregs[list_num])))
    print(''+str(intersection))


# In[122]:


for list_num in range(6):
    intersection = list(set(map(tuple, fp_data_downregs[list_num])) & set(map(tuple, b_data_downregs[list_num])))
    print(intersection)


# In[ ]:





# In[ ]:




