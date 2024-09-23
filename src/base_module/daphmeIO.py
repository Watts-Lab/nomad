import constants
import pyarrow.parquet as pq
import pandas as pd
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
    def __init__(self, labels = {}, session=None):
        self.schema = constants.DEFAULT_SCHEMA
        self.update_schema(labels)
        self.df = None
        self.session = session
        
    def update_schema(self, labels: list[str]) -> None:
        for label in labels:
            if label in self.schema:
                self.schema[label] = labels[label]

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

    


