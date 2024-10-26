#!/home/lic/pmhome-py/bin/python
#<--- import required modules --->#
from sqlalchemy import create_engine, MetaData, Table, select, insert, values
import psycopg2


#<---- user's definitions ---->#
from sqlalchemy import create_engine, Table, MetaData

def insert_dataframe(db_url='postgresql+psycopg2://postgres:sqldbpass@127.0.0.1:5432/licdb',
                     db_table='pqube_fi', 
                     data_frame={},
                     debug=False):
    """
    NAME
        insert_dataframe(args)
    DESCRIPTION
        This function uses sqlalchemy module to append a dataframe (data_frame)
        to a target table (db_table) inside the connected database (db_url). 
    ARGUMENTS
        db_url (str): Database URL. URLs typically include username, password, hostname and database name fields
        db_table (str): Name of the table inside the target database
        data_frame (dict): Dictionary with pair of table id headers and data to insert
        debug (bool): echo sqlalchemy module and database messages for debugging
    RETURNS
    """
    engine = create_engine(db_url, echo=debug)
    try:
        with engine.connect() as conn:  # type:ignore
            metadata = MetaData()
            tabla = Table(db_table, metadata, autoload_with=engine)  # Usa solo autoload_with
            conn.execute(tabla.insert(), data_frame)
            print("Dataframe inserted in table named " + db_table)
    except Exception as e:  # Captura la excepción
        print("Error raised when writing table:", e)  # Imprime el mensaje de error

# no está reconociendo el argumento autoload. Este argumento fue reemplazado en versiones más recientes de SQLAlchemy.