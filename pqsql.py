#<--- import required modules --->#
import sqlalchemy as db 
import psycopg2



#<---- user's definitions ---->#
def fetch_last_n_rows( db_url='postgresql+psycopg2://postgres:sqldbpass@10.2.11.68:5432/licdb',
                       db_table='pqube_fi',
                       nrows=1):
    
    """ 
    NAME
        fetch_last_n_rows(args)
    DESCRIPTION
        This function uses sqlalchemy module to read the last n-rows (nrows) off a 
        target table (db_table). The information for connecting to the
        target SQL server and database is parsed in url format (db_url).
    ARGUMENTS
        db_url (str): Database URL. URLs typically include username, password, hostname and database name fields
        db_table (str): Name of the table inside de target database
        nrows (int): Number of rows to be read 
    RETURNS
        data(list): return a list of list size [nrows][all collumns in table]
    """
    engine   = db.create_engine(db_url)
    metadata = db.MetaData()
    tabla    = db.Table(db_table, metadata, autoload_with=engine)
    stmt     = db.select(tabla.columns).order_by(tabla.columns.time.desc()).limit(nrows)
    try:
        with engine.connect() as conn:
            data = conn.execute(stmt).fetchall()
    except:
        print("Error raised when reading table")
        data = []
    return data     


def fetch_one_column_last_n_rows(db_url='postgresql+psycopg2://postgres:sqldbpass@10.2.11.68:5432/licdb',
                      db_table='pqube_fi',
                      column_label='time',
                      nrows=1):  
    """ 
    NAME
    DESCRIPTION
    ARGUMENTS
        db_url (str): Database URL. URLs typically include username, password, hostname and database name fields
        db_table (str): Name of the table inside de target database
        nrows (int): Number of rows to be read 
    RETURNS
    """
    engine   = db.create_engine(db_url)
    metadata = db.MetaData()
    tabla    = db.Table(db_table, metadata, autoload_with=engine)
    stmt     = db.select(tabla.columns[column_label]).order_by(tabla.columns.time.desc()).limit(nrows)
    try:
        with engine.connect() as conn:
            result = conn.execute(stmt).fetchall()
            data = list(zip(*result))[0]
    except:
        print("Error raised when reading table")
        data = []
    return data


def get_column_labels(db_url='postgresql+psycopg2://postgres:sqldbpass@10.2.11.68:5432/licdb',
                      db_table='pqube_fi'):

    engine   = db.create_engine(db_url)
    metadata = db.MetaData()
    tabla    = db.Table(db_table, metadata, autoload_with=engine)
    stmt     = db.select(tabla)
    try:
        with engine.connect() as conn:
            data = conn.execute(stmt)

        for key in data.keys():
            print(key)
    except:
        print("Error raised when reading table") 
        data = []        
    return data