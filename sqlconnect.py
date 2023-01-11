import pandas as pd
import pyodbc
from sqlalchemy.engine import URL
from sqlalchemy import create_engine

connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.10.53.68;DATABASE=FORECAST_DB;UID=sa;PWD=axali1Paroli;'


def getdatafromsql():
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

    engine = create_engine(connection_url)

    sql_query = pd.read_sql(
        "SELECT * FROM [FORECAST_DB].[dbo].[vwTEST_DNN] where c+cy+c7 is not null order by DT",
        engine)

    df = pd.DataFrame(sql_query)
    df.index = pd.to_datetime(df ['DT'])

    sql_query = pd.read_sql(
        "SELECT * FROM [FORECAST_DB].[dbo].[vwTEST_DNN] where c is null and dt>'2022-12-01' order by DT",
        engine)

    df1 = pd.DataFrame(sql_query)
    df1.index = pd.to_datetime(df1 ['DT'])

    return df, df1


def getdatafromsqlols():
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

    engine = create_engine(connection_url)

    sql_query = pd.read_sql(
        "select * from vwTEST where c+cy+c7 is not null and date>='2017-06-01'order by date",
        engine)

    df = pd.DataFrame(sql_query)
    df.index = pd.to_datetime(df ['Date'])

    sql_query = pd.read_sql(
        "select * from vwTEST where c is null and date>'2022-12-01' order by date",
        engine)

    df1 = pd.DataFrame(sql_query)
    df1.index = pd.to_datetime(df1 ['Date'])

    return df, df1


def insert_model(model_id, model, json_data, r2, loss, val_loss, mape, val_mape, elapsed):
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

    engine = create_engine(connection_url)
    insert = 'INSERT INTO dbo.BuiltModels(BuildDay,BuildDate,ModelID,ModelDump,ModelJson,R2,loss,val_loss,mape,val_mape,elapsed) VALUES (getdate(),getdate(),?,?,?,?,?,?,?,?,?)'

    params = (model_id, pyodbc.Binary(model), json_data, r2, loss, val_loss, mape, val_mape, elapsed)
    with engine.begin() as connection:
        connection.execute(insert, params)

