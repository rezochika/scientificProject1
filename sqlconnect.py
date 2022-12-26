import pandas as pd
from sqlalchemy.engine import URL
from sqlalchemy import create_engine


def getdatafromsql():
    connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.10.53.68;DATABASE=FORECAST_DB;UID=sa;PWD=axali1Paroli;'
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

    engine = create_engine(connection_url)

    sql_query = pd.read_sql(
        "SELECT DT,wf,mf,sf,IHT,IH,wfy,sqhdd1,dhdd1,dhdd13,hdd1t,Troloff,mx2,mn4,dcy,c7,cy,c FROM [FORECAST_DB].[dbo].[vwTEST_DNN] where c+cy+c7 is not null order by DT",
        engine)

    df = pd.DataFrame(sql_query)
    df.index = pd.to_datetime(df['DT'])

    sql_query = pd.read_sql(
        "SELECT DT,wf,mf,sf,IHT,IH,wfy,sqhdd1,dhdd1,dhdd13,hdd1t,Troloff,mx2,mn4,dcy,c7,cy,c FROM [FORECAST_DB].[dbo].[vwTEST_DNN] where c is null and dt>'2022-12-01' order by DT",
        engine)

    df1 = pd.DataFrame(sql_query)
    df1.index = pd.to_datetime(df1['DT'])

    return df, df1
