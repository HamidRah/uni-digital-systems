import mysql.connector
from mysql.connector import errorcode

# MYSQL CONFIG VARIABLES
hostname    = "db"
username    = "hamid"
passwd  = "U9kknW3Q^8"
db = "moviemanager_db"

def get_db_connection():
    try:
        print('Connecting to the MySQL database...')
        conn = mysql.connector.connect(host=hostname,                              
                              user=username,
                              password=passwd,
                              database=db)  
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print('User name or Password is not working')
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print('Database does not exist')
        else:
            print(err)                        
    else:  #will execute if there is no exception raised in try block
        return conn   