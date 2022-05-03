#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data Preprocessing


# In[2]:


import pandas as pd
import numpy as np
import os
import glob
import biomart 


# In[3]:


def read_df(filepath,filename):
    df1 = pd.read_csv(filepath,header=None, sep= '\t',skiprows=6,index_col=0)
    df1
    df1 = df1[[6]]
    df1 = df1.T
    df1.index = [filename]
    return df1


# In[4]:


def get_ensembl_mappings():                                   
    # Set up connection to server                                               
    server = biomart.BiomartServer('http://uswest.ensembl.org/biomart')         
    mart = server.datasets['hsapiens_gene_ensembl']                            
                                                                                
    # List the types of data we want                                            
    attributes = ['ensembl_transcript_id','external_gene_name', 
                  'ensembl_gene_id', 'ensembl_peptide_id']
                                                                                
    # Get the mapping between the attributes                                    
    response = mart.search({'attributes': attributes})                          
    data = response.raw.data.decode('ascii')                                    
                                                                                
    ensembl_to_genesymbol = {}                                                  
    # Store the data in a dict                                                  
    for line in data.splitlines():                                              
        line = line.split('\t')                                                 
        # The entries are in the same order as in the `attributes` variable
        transcript_id = line[0]                                                 
        gene_symbol = line[1]                                                   
        ensembl_gene = line[2]                                                  
        ensembl_peptide = line[3]                                               
                                                                                
        # Some of these keys may be an empty string. If you want, you can 
        # avoid having a '' key in your dict by ensuring the 
        # transcript/gene/peptide ids have a nonzero length before
        # adding them to the dict
        #ensembl_to_genesymbol[transcript_id] = gene_symbol                      
        ensembl_to_genesymbol[ensembl_gene] = gene_symbol                       
        #ensembl_to_genesymbol[ensembl_peptide] = gene_symbol                
                                                                                
    return ensembl_to_genesymbol


# In[5]:


def get_key(mappings,val):
    for key,valu in mappings.items():
        if valu == val:
            return key
        
    return None


# In[6]:


#Reading Sample_sheet
sample_data = pd.read_csv("./Addiotional_Data/metadata/gdc_sample_sheet.2022-04-26.tsv", delimiter='\t') #Reading Sample Data
sample_data['filename_short'] = sample_data['File Name'].apply(lambda x: x.split('.')[0]) #Stripping File name (Removing FPKM.txt.gz)


# In[7]:


#Reading Clinical Data to get labels
clinical_data = pd.read_csv("./Addiotional_Data/metadata/Clinical/clinical.tsv", delimiter='\t')
clinical_data = clinical_data[['case_submitter_id','primary_diagnosis']]
u = clinical_data.groupby("case_submitter_id").agg(list).reset_index() #Grouping all samples and their diagnosis to a list
u['len'] = u['primary_diagnosis'].apply(lambda x: len(x)) #Getting length of each list to check how many times samples have their clinical data
u['all_equal'] = u['primary_diagnosis'].apply(lambda x: len(set(x))) #Checking if every time all the entries added in primary tumor are the same
u['final_label'] = u['primary_diagnosis'].apply(lambda x: x[0]) #creating the final label for the dataset


# In[8]:


#removing mixed gliomas, gliosarcoma and merging Astrocytomas and oligodendroglioma together
dict_map = {"Astrocytoma, NOS":"Astrocytoma", "Astrocytoma, anaplastic": "Astrocytoma",
            "Oligodendroglioma, NOS":"Oligodendroglioma","Oligodendroglioma, anaplastic": "Oligodendroglioma",
            "Glioblastoma":"Glioblastoma","Gliosarcoma":"Gliosarcoma"}

sample_data = sample_data.merge(u[['case_submitter_id','final_label']], how='left', left_on='Case ID', right_on='case_submitter_id')
sample_data = sample_data.dropna()

label_df = sample_data[['File Name','Project ID','Case ID','final_label']]
label_df
dropped_files = set()
dropped_files.update(label_df[label_df["final_label"].isna()]["File Name"])
label_df = label_df[~label_df["final_label"].isna()].reset_index(drop = True)
dropped_files.update(label_df[label_df["final_label"] == 'Gliosarcoma']["File Name"])
label_df = label_df[label_df["final_label"] != 'Gliosarcoma'].reset_index(drop = True)
dropped_files.update(label_df[label_df["final_label"] == 'Mixed glioma']["File Name"])
label_df = label_df[label_df["final_label"] != 'Mixed glioma'].reset_index(drop = True)
label_df["labelf"] = label_df["final_label"].apply(lambda x: dict_map[x])
label_df


# In[9]:


#Reading All the data Files
rootdir = './data/'

filelist = glob.glob("./data/*/*.tsv") #Getting path to all files, using the glob module
filelist = [(x,x.split('/')[3]) for x in filelist] #creates a tuple (x,y) x = path to file, y = name of file for accession


df_list = []

for x in filelist:
    tmp_df = read_df(x[0],x[1])
    df_list.append(tmp_df)
    
master_df = pd.concat(df_list)
master_df


# In[10]:


#Selecting Data from Label
data_df = master_df
data_df = data_df[data_df.index.isin(label_df["File Name"])]
data_df


# In[11]:


#Removing zero columns
zero_cols = []
all_cols = data_df.columns



for col in data_df.columns:
    if data_df[col].sum()==0:
        zero_cols.append(col)
        
remaining_cols = list(set(all_cols) - set(zero_cols))
data_df = data_df[remaining_cols]
data_df


# In[12]:


#Removing all Zero in per particular project
project_ids = set(label_df['Project ID'])

project_dict = {}

for project in project_ids:
    file_names = list(label_df[label_df['Project ID'] == project]['File Name'])
    project_dict[project] = file_names
    
zero_cols = []
all_cols = data_df.columns

for key,val in project_dict.items():
    df = data_df[data_df.index.isin(val)]
    for col in df.columns:
        if df[col].sum() == 0:
            zero_cols.append(col)
    
remaining_cols = list(set(all_cols) - set(zero_cols))
data_df = data_df[remaining_cols]
data_df


# In[13]:


# Top 300 Genes
df1 = pd.read_csv('./Addiotional_Data/selected_genes/genes_0-100.tsv', sep ='\t')
df2 = pd.read_csv('./Addiotional_Data/selected_genes/genes_100-200.tsv', sep ='\t')
df3 = pd.read_csv('./Addiotional_Data/selected_genes/genes_200-300.tsv', sep ='\t')

top_genes_df = pd.concat([df1,df2,df3],ignore_index=True)
top_genes_df


# In[14]:


#Converting Symbol to Ensemble Gene_IDS
mappings = get_ensembl_mappings()


# In[15]:


top_genes_df['ensemble'] = top_genes_df.apply(lambda x: get_key(mappings,x['Symbol']).strip(), axis = 1)
top_genes_df


# In[16]:


cols = [x.split('.')[0].strip() for x in data_df.columns]
cols

selected_cols = set.intersection(set(cols),set(top_genes_df['ensemble']))


# In[17]:


data_df.columns = cols
gene_filtered_df = data_df[list(selected_cols)]
gene_filtered_df = gene_filtered_df.apply(lambda x: np.log2(x+0.001), axis = 1)
data_df = gene_filtered_df


# In[ ]:





# In[18]:


#Reading Normal Data
normal_df_selected_500 = pd.read_csv("./Addiotional_Data/controls/normal_500.txt",index_col=0)
normal_df_selected_200 = pd.read_csv("./Addiotional_Data/controls/normal_200.txt",index_col=0)
# In[19]:


labeldf = label_df[['File Name','labelf']]


# In[20]:


#MakingDataset for Glioblastoma vs Astrocytoma vs Oligodendroglioma
os.system('mkdir ./output/data_preprocessing/GvsAvsO')
data_df = data_df.loc[labeldf['File Name']]
data_df.to_csv("./output/data_preprocessing/GvsAvsO/X.txt")
labeldf.to_csv("./output/data_preprocessing/GvsAvsO/y.txt")


# In[21]:


#MakingDataset for Glioblastoma vs (A + O)
os.system('mkdir ./output/data_preprocessing/GvsAO')
mapping = {'Glioblastoma':'Glioblastoma','Normal':'Normal','Astrocytoma':'A+O','Oligodendroglioma':'A+O'}
labeldf_2 = labeldf.copy()
labeldf_2['labelf'] = labeldf_2['labelf'].apply(lambda x: mapping[x])
data_df = data_df.loc[labeldf_2['File Name']]
data_df.to_csv("./output/data_preprocessing/GvsAO/X.txt")
labeldf_2.to_csv("./output/data_preprocessing/GvsAO/y.txt")


# In[22]:


#MakingDataset for Glioblastoma vs Astrocytoma vs Oligodendroglioma vs Normal
data_all = pd.concat([data_df,normal_df_selected_200])

#Making normal labels and adding it to labels
norm_lab = [(x,'Normal') for x in normal_df_selected_200.index]
norm_lab_df = pd.DataFrame(norm_lab, columns=['File Name','labelf'])
labeldf_3 = pd.concat([labeldf,norm_lab_df]).reset_index(drop = True)

data_all = data_all.loc[labeldf_3['File Name']] # making sure the order is correct

os.system("mkdir ./output/data_preprocessing/GvsAvsOvsN")
data_all.to_csv("./output/data_preprocessing/GvsAvsOvsN/X.txt")
labeldf_3.to_csv("./output/data_preprocessing/GvsAvsOvsN/y.txt")


# In[25]:


#MakingDataset for Glioblastoma vs (A+O) vs Normal
labeldf_4 = pd.concat([labeldf_2,norm_lab_df]).reset_index(drop = True) #merging G vs (A+O) with Normal
data_all = data_all.loc[labeldf_4['File Name']]
os.system("mkdir ./output/data_preprocessing/GvsAOvsN")
data_all.to_csv("./output/data_preprocessing/GvsAOvsN/X.txt")
labeldf_4.to_csv("./output/data_preprocessing/GvsAOvsN/y.txt")


# In[27]:


#MakingDataset for Glioma vs Normal
mapping = {'Glioblastoma':'Glioma','Normal':'Normal','Astrocytoma':'Glioma','Oligodendroglioma':'Glioma'}
data_all = pd.concat([data_df,normal_df_selected_500])
#Making normal labels and adding it to labels
norm_lab = [(x,'Normal') for x in normal_df_selected_500.index]
norm_lab_df = pd.DataFrame(norm_lab, columns=['File Name','labelf'])
labeldf_5 = pd.concat([labeldf,norm_lab_df]).reset_index(drop = True) #Adding normal label to label_df
labeldf_5['labelf'] = labeldf_5['labelf'].apply(lambda x: mapping[x]) #converting all gliomas
data_all = data_all.loc[labeldf_5['File Name']]

os.system("mkdir ./output/data_preprocessing/GvsN")
data_all.to_csv("./output/data_preprocessing/GvsN/X.txt")
labeldf_5.to_csv("./output/data_preprocessing/GvsN/y.txt")


# In[34]:


#MakingDataset for A vs O
labeldf_6 = labeldf[labeldf['labelf'].isin(['Oligodendroglioma','Astrocytoma'])].reset_index(drop = True)
data_all = data_df.loc[labeldf_6['File Name']]

os.system("mkdir ./output/data_preprocessing/AvsO")
data_all.to_csv("./output/data_preprocessing/AvsO/X.txt")
labeldf_6.to_csv("./output/data_preprocessing/AvsO/y.txt")

