# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:13:45 2020
"""

"""
Modifications copyright (C) 2021 Hristo Petkov
This file is not part of the original DAG-GNN code.
Modifications are as follows:
  -Addition of a new file containing pre-processing functionality for the input data
"""

"""
@inproceedings{xu2019modeling,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}

@article{torfi2020cor,
title={COR-GAN: Correlation-Capturing Convolutional Neural Networks for Generating Synthetic Healthcare Records},
author={Torfi, Amirsina and Fox, Edward A},
journal={arXiv preprint arXiv:2001.09346},
year={2020}
}
"""

#Importing libraries and frameworks
import os
import numpy as np
import torch
import pandas as pd
from ctgan.data import read_csv

class FullDataPreProcessor:
    
    def __init__(self, path, column_names, initial_identifier, num_of_rows, seed):
        self.path = path
        self.column_names = column_names
        self.initial_identifier = initial_identifier
        self.num_of_rows = num_of_rows
        self.seed = seed
    
    def get_dataframe(self):
        
        df = read_csv(self.path)
        
        #Getting all of the columns with regards to their dtype
        non_numeric_columns = list(df[0].select_dtypes(exclude=['int64','float64']).columns)
        numeric_int_columns = list(df[0].select_dtypes(include=['int64']).columns)
        numeric_float_columns = list(df[0].select_dtypes(include=['float64']).columns)
            
        #Filling in all of the missing data of type string
        for j in range(len(non_numeric_columns)):
            df[0][non_numeric_columns[j]].fillna('emptyblock', inplace = True)
            
        #Filling in all of the missing data of type int
        for k in range(len(numeric_int_columns)):
            df[0][numeric_int_columns[k]].fillna(-123456789, inplace = True)
             
        #Filling in all of the missing data of type float    
        for l in range(len(numeric_float_columns)):
            df[0][numeric_float_columns[l]].fillna(-1234.56789, inplace = True)
        
        return df
    
    def sample_dataframe(self, dataframe): 
        if self.column_names != []:
            dataframes = []
            #assert self.initial_identifier != '', 'Initial Identifier not specified! Choose one of the following: ' + str(list(dataframe.columns))
            #initial_df = pd.DataFrame({self.initial_identifier: dataframe[self.initial_identifier]})
            #dataframes.append(initial_df)
            for column in self.column_names:
                tmpdf = pd.DataFrame({column: dataframe[column]})
                dataframes.append(tmpdf)
            new_df = pd.concat(dataframes, axis=1)
            if self.num_of_rows != -1:
                assert self.num_of_rows > 0, 'Number of rows must be greater than zero'
                assert self.num_of_rows <= dataframe.shape[0], 'Number of rows must less or equal to the total number of rows'
                sampled_df = new_df.sample(self.num_of_rows, random_state=self.seed)
                sampled_df.sort_index(inplace=True)
                return sampled_df
            else:
                return new_df
        else:
            return dataframe
    
class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data here
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        if self.transform:
           pass

        return torch.from_numpy(sample)
         
