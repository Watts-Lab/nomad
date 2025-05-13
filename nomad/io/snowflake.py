def get_table_head(snow_engine, 
                   table_name, 
                   n_rows = 5): 
    '''
    get the first rows of a sql table as a pandas dataframe
    '''
    
    query = f"SELECT * FROM {table_name} LIMIT {n_rows}"
    df_head = snow_engine.read_sql(query)
    
    return df_head


def table_column_unique(snow_engine, 
                            table_name, 
                            column):
    """
    Get the list of unique values from a specific column in an SQL table.
    """
    query = f"SELECT DISTINCT {column} FROM {table_name}"
    df_unique = snow_engine.read_sql(query)
    
    return df_unique[column].dropna().tolist()

def table_column_filtered(snow_engine, 
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

def table_columns(snow_engine, table_name, include_schema=False):
    """
    Either return the column names or, if include_schema is True, the full table schema.

    Parameters
    ----------
    snow_engine : sqlalchemy.engine.Engine
        A live Snowflake engine.
    table_name : str
        The (optionally database- and schema-qualified) table name.
    include_schema : bool, default False
        When False (default) the function returns the column names;
        when True it returns the result of `DESCRIBE TABLE`, i.e. the schema.

    Returns
    -------
    pandas.Index | list[str] | pandas.DataFrame
        Column names as a pandas Index (or list) or, if requested, a DataFrame
        with the schema.
    """
    if include_schema:
        query = f"DESCRIBE TABLE {table_name}"
        return snow_engine.read_sql(query)
    query = f"SELECT * FROM {table_name} LIMIT 0"
    return snow_engine.read_sql(query).columns

