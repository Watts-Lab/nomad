import constants
import pyarrow.parquet as pq
import pandas as pd
import os
from functools import partial
import multiprocessing
from multiprocessing import Pool
import re
from pyspark.sql import SparkSession

def get_pq_users(path: str, id_string: str):
    return pq.read_table(path, columns=[id_string]).column(id_string).unique().to_pandas()

def get_pq_user_data(path: str, users: list[str], id_string: str):
    return pq.read_table(path, filters=[(id_string, 'in', users)]).to_pandas()

class DataLoader():
    def __init__(self, data_path: str, labels = {}, session=None, file_format = None):
     
        self.labels = constants.DEFAULT_LABELS
        self.update_labels(labels)
        self.data_path = data_path
        self.session = session

        #data fields
        self.df_all = None
        self.users = None

        if (file_format is not None):
            self.format = file_format
        else:
            self.format = None
            if(os.path.isdir(self.data_path)):
                for root, dirs, files in os.walk(self.data_path, topdown=True):
                    for name in files:
                        if(name.endswith('.csv')):
                            self.format = 'csv'
                            break
                        elif(name.endswith('.parquet')):
                            self.format = 'parquet'
                            break
                    if(self.format is not None):
                        break
            else:
                if(data_path.endswith('.csv')):
                    self.data_path = 'csv'
                elif(data_path.endswith('.parquet')):
                    self.data_path = 'parquet'
        if(self.format is None):
            raise Exception('No valid data format provided nor found in path.')
    
    def update_labels(self, labels: list[str]) -> None:
        """
        Updates map for custom column headers
        Args:
            labels (list[str]): a dictionary of the default labels to the custom ones
        """        
        for label in labels:
            if label in self.labels:
                self.labels[label] = labels[label]
        
    def get_users(self) -> None:
        """
        Get all the users from the data path as an array under users
        """        
        if(self.df_all is not None):
            self.users = self.df_all[self.labels['uid']].unique()
        else:
            match self.format:
                case 'parquet':
                    self.users = pq.read_table(self.data_path, columns=[self.labels['uid']]).column(self.labels['uid']).unique().to_pandas().values
                case 'csv':
                    self.get_all()
                    self.users = self.df_all[self.labels['uid']].unique()

    def get_all(self) -> None:
        """
        Get all data from the data path under as a pandas dataframe under df_all
        """        
        match self.format:
            case 'parquet':
                self.df_all = pq.read_table(self.data_path).to_pandas()
            case 'csv':
                if(os.path.isdir(self.data_path)):
                    csvs = []
                    for root, dirs, files in os.walk(self.data_path, topdown=False):
                        for name in files:
                            if(name.endswith('.csv')):
                                csvs.append(pd.read_csv(os.path.join(root, name), index_col=0))       
                    self.df_all = pd.concat(csvs)
                else:
                    self.df_all = pd.read_csv(self.data_path, index_col=0)


    # def load_gravy_sample(self, paths, user_count, cpu_count = multiprocessing.cpu_count()):
    #     self.update_schema(constants.GRAVY_SCHEMA)
    #     users = []
    #     with Pool(cpu_count) as p:
    #         users.extend(p.map(partial(get_pq_users, id_string=self.schema['id']), paths))
    #     all_users = pd.concat(users).drop_duplicates()
    #     all_users = all_users.rename(self.schema['id'])
    #     all_users = all_users.apply(str)
    #     all_users = all_users.sample(user_count)

    #     data = []
    #     with Pool(multiprocessing.cpu_count()) as p:
    #         data.extend(p.map(partial(get_pq_user_data, users=all_users, id_string=self.schema['id']), paths))
    #     self.df = pd.concat(data).drop_duplicates()

    
    #SPARK STUFF:


    def link_session(self, session) -> None:
        self.session = session

    def config_s3(self) -> None:
        self.session\
        .config("spark.jars.packages", "org.apache.spark:spark-hadoop-cloud_2.12:3.3.0")\
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")\
    
    def load_s3(self, path: list[str]) -> None:
        if(self.session == None):
            raise Exception("No Session Initiated")
        if(path.startswith('s3:')):
            load_path = 's3a:'+path[3:]
        else:
            load_path = path
        self.df = self.session.read.parquet(load_path)
        
    

    


