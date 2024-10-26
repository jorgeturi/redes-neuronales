from sqlalchemy import create_engine, inspect

# Conectar a la base de datos SQLite
engine = create_engine('sqlite:///datacamp.sqlite')  # Asegúrate de usar la ruta correcta de tu base de datos

# Usar el inspector de SQLAlchemy para obtener la estructura de la base de datos
inspector = inspect(engine)

# Obtener todas las tablas de la base de datos
tables = inspector.get_table_names()

# Mostrar las tablas
print("Tablas en la base de datos:")
for table_name in tables:
    print(f"- {table_name}")
    
    # Obtener las columnas de la tabla
    columns = inspector.get_columns(table_name)
    
    # Mostrar las columnas de cada tabla
    print(f"  Columnas de la tabla '{table_name}':")
    for column in columns:
        print(f"    Nombre: {column['name']}, Tipo: {column['type']}")