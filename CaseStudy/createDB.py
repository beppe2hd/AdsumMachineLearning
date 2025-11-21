import pandas as pd



import pandas as pd
import mysql.connector
from mysql.connector import Error

def save_dataset_to_mysql(df, host, user, password, database, table_name="data_table"):
    """
    Saves a pandas DataFrame with columns ['datetime', 's_a', 's_b', 'irr', 'LAI'] into a MySQL table.
    """
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        if conn.is_connected():
            cursor = conn.cursor()

            # Create table if not exists
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                datetime DATETIME NOT NULL,
                s_a FLOAT,
                s_b FLOAT,
                irr FLOAT,
                LAI FLOAT
            );
            """
            cursor.execute(create_table_query)

            # Insert dataset rows
            insert_query = f"""
            INSERT INTO {table_name} (datetime, s_a, s_b, irr, LAI)
            VALUES (%s, %s, %s, %s, %s);
            """

            # Convert DataFrame to list of tuples
            data_tuples = [tuple(x) for x in df.to_numpy()]
            cursor.executemany(insert_query, data_tuples)

            conn.commit()
            print(f"Successfully inserted {cursor.rowcount} rows into {table_name}.")

    except Error as e:
        print(f"Error: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- Example usage ---
if __name__ == "__main__":

    df = pd.read_csv("field_f6")
    df = df[['datetime','s_a','s_b','irr','LAI']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'] + pd.DateOffset(months=9)

    # Save to database
    save_dataset_to_mysql(
        df,
        host="localhost",
        user="root",
        password="password",
        database="sensors",
        table_name="sensor_data"
    )



