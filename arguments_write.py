#!/home/lic/pyvenv/bin/python
#<--- main project constants --->#
prediction_window = 1

#<--- dataframe dictionary--->#
dataframe = {'next_time': str(), 'next_power_watt': 0.0}


#<--- postgreSQL constants--->#
PSQL_HOST           = "10.2.11.68"
PSQL_PORT           = "5432"
PSQL_USER           = "postgres"
PSQL_PASSWORD       = "sqldbpass"
PSQL_DB_NAME        = "licdb"
PSQL_TABLE          = "forecasting"
PSQL_DIALECT        = "postgresql"
PSQL_DRIVER         = "psycopg2"
PSQL_ENGINE_URL     = "{}+{}://{}:{}@{}:{}/{}".format(PSQL_DIALECT, 
                                                      PSQL_DRIVER, 
                                                      PSQL_USER, 
                                                      PSQL_PASSWORD, 
                                                      PSQL_HOST, 
                                                      PSQL_PORT, 
                                                      PSQL_DB_NAME)