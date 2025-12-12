"""
A module to connect to the 'airflow' PostgreSQL database and fetch data using a provided SQL query.

Usage example:
    >>> from scripts import actidb
    >>> QUERY = "SELECT * FROM muestra.nvda nvda ORDER BY nvda.date;"
    >>> nvda = actidb.fetch(QUERY)
"""

import psycopg2
import pandas as pd

def fetch(QUERY):
    hostname = 'localhost'
    database = 'airflow'
    username = 'airflow'
    pwd = 'airflow'
    port_id = 5433
    # Conn es un objeto de conexión a la base de datos
    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=pwd,
            port=port_id
        )
        print("Conexión exitosa")

        result = pd.read_sql(QUERY, conn)

        conn.close()
        print("Conexión cerrada")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

    return result