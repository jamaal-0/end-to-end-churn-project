import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="churn_db",
        user="postgres",
        password="tarantinoV_1",
        port=5432
    )