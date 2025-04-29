def get_table_head(snow_engine, 
                   table_name, 
                   n_rows = 5): 
    '''
    get the first rows of an sql table as a pandas dataframe
    '''
    
    query = f"SELECT * FROM {table_name} LIMIT {n_rows}"
    df_head = snow_engine.read_sql(query)
    
    return df_head


def get_table_column_unique(snow_engine, 
                            table_name, 
                            column):
    """
    Get the list of unique values from a specific column in an SQL table.
    """
    query = f"SELECT DISTINCT {column} FROM {table_name}"
    df_unique = snow_engine.read_sql(query)
    
    return df_unique[column].dropna().tolist()

def get_table_column_filtered(snow_engine, 
                              table_name, 
                              column, 
                              value):
    """
    Get the dataframe from an SQL table where a specific column equals a certain value.
    """
    
    query = f"""SELECT * FROM {table_name} 
                WHERE {column} = '{value}'
             """
    
    df_filtered = snow_engine.read_sql(query)
    
    return df_filtered