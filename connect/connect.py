import pyodbc
from pandas import read_sql  
from configparser import ConfigParser
from sqlalchemy import create_engine
import ast
import getpass
import os
import warnings


class EstablishConnection:
    def __init__(self, conn_type, instance, database):        

        self._conn_type = conn_type
        self._df = None
        self._last_columns = None
        self.ref_subquery = None

        # initiating config parser
        self.config = ConfigParser()
        user_config = ConfigParser()

        # setting root configuration path
        ## maintains user credentials secret
        ## only accessible by user
        system_driver = os.environ['SYSTEMDRIVE']        
        user_name = getpass.getuser()
        password = 'placeholder'
        has_password = False
        has_user = False
        user_path = f"{system_driver}/Users/{user_name}/.config"
        user_full_path = os.path.join(user_path, 'user.ini')
        self._validating_secret_user_cred(user_path, user_full_path)
        user_config.read(user_full_path)

        ## maitain general server info under package path
        ## this is accessible by all users
        script_path = os.path.realpath(__file__)
        root_path = '\\'.join(script_path.split('\\')[:3])
        self.root_path = root_path

        # accessing .config/alvn.ini file
        self._validating_config_folder()
        full_path = os.path.join(root_path, 'avln.ini')
        self.full_path = full_path

        # # check if the file exist or have data        
        self._validating_config_file()
        self.config.read(full_path)
        sections = self.config.sections()
        
        # extracting connection information
        # this is a dictionary of server and connection
        # string information
        err = None
        try:
            conn_type_func = lambda sec: conn_type.lower() in sec
            conn_found = list(filter(conn_type_func, sections))[0]
            self._conn_type = conn_found
            conn_info_dic = self.config[conn_found]

            # getting user cred
            if conn_type == 'redshift':
                
                # configuring user name
                user_name = user_config.get(conn_found, 'user')
                if user_name == '':
                    user_name = input("Username: ")
                else:
                    has_user = False
                
                # configuring password
                password = user_config.get(conn_found, 'password')
                if password == '':
                    password = getpass.getpass("Password: ")
                    save_credentials = input("Do you want to save your credentials [y/n]")

                    if save_credentials.lower() in ['yes', 'y']:
                        user_config.set('aws_redshift', 'user', user_name)
                        user_config.set('aws_redshift', 'password', password.replace('%', '%%'))
                        with open(user_full_path, 'w+') as configfile:
                            user_config.write(configfile)
                        has_user = True
                        has_password = True

                else:
                    has_password = True                

            # extracting server information
            # this is a dictionary of server information
            driver = conn_info_dic['driver']
            servers = ast.literal_eval(conn_info_dic['servers'])
            err = 'server'
            server_func = lambda sv: instance.lower() in sv.lower()
            server_found = list(filter(server_func, servers))[0]

            # extracting instance information such 
            # as endpoint and port
            instance_info_dic = servers[server_found]

            # extracting connection string information
            conn_string = ast.literal_eval(conn_info_dic['connection_strings'])

            # extracting attributes
            endpoint = instance_info_dic.get('endpoint')
            port = instance_info_dic.get('port')
            pyodbc_string = conn_string['pyodbc']
            alchemy_string = conn_string['alchemy']

            # access dictionary
            err = 'database'
            access_cred = dict(
                endpoint=endpoint,
                database=database,
                driver=driver,
                user=user_name,
                password=password,
                port=port)

            # creating connection strings
            pyodbc_string = pyodbc_string.format(**access_cred)
            self.alchemy_string = alchemy_string.format(**access_cred).replace(' ', '+')

            # establishing connection
            self.conn = pyodbc.connect(pyodbc_string)
            self.cursor = self.conn.cursor()
            self.commit = self.conn.commit
            self.rollback = self.conn.rollback
            self.close = self.conn.close

        except Exception as e:

            if err == 'server':
                raise ValueError(f"Instance not found {instance}. "
                    f"Available instances {servers}")
            else:
                if has_user or has_password:
                    
                    warnings.warn("Username and password failed. Update credentials and try again.")
                    user = input("New username: ")
                    user_config.set('aws_redshift', 'user', user)
                    password = getpass.getpass("Password: ")
                    password = password.replace('%', '%%')
                    user_config.set('aws_redshift', 'password', password)
                    with open(user_full_path, 'w+') as configfile:
                        user_config.write(configfile)

                    self.__init__(conn_type, instance, database)

                raise ValueError("Connection, database, or credentials failed!")

    def read(self, query):  
        self._df = read_sql(query, self.conn)
        self._last_columns = self._df.columns.tolist()
        return self._df

    def run_sql(self, query, show_query=False):
        
        # replacing the reference_table
        if self.ref_subquery:
            query = query.replace('ref_table', f'({self.ref_subquery}) as ref_table')
        
        if show_query: print(query)

        # running the query
        self._df = read_sql(query, self.conn)
        self._last_columns = self._df.columns.tolist()
        return self._df

    def insert_frame(self, dataframe, schema=None, table_name=None, truncate=False, fast=False):
        cursor = self.cursor

        if schema == None and table_name == None:
            raise ValueError("Please provide schema and table_name. You can provide both in the table_name argument as dbo.table.")

        # infer table if it is given as one
        if str(schema).find('.') > 0: 
            split = schema.split('.')
            schema = split[0]
            table_name = split[1]
        if str(table_name).find('.') > 0:
            split = table_name.split('.')
            schema = split[0]
            table_name = split[1]

        # get table name and perpare insert statement
        table = f"{schema}.{table_name}"
        insert_to_tmp_tbl_stmt = f"INSERT INTO {table}({', '.join(dataframe.columns)}) VALUES ({','.join(['?']*dataframe.shape[1])})"

        # truncate table before insert if user states it
        if truncate: cursor.execute(f"TRUNCATE TABLE {table}")

        # insert data from the dataframe and commit
        if fast: cursor.fast_executemany = True
        cursor.executemany(insert_to_tmp_tbl_stmt, dataframe.values.tolist())
        cursor.commit()

        print(f'{len(dataframe)} rows inserted to the {table} table')
    
    def createOrReplaceSubRef(self, subquery):
        self.ref_subquery= subquery

    def show_views(self):
        self.read("""
        select table_schema as schema_name,
            table_name as view_name,
            view_definition
        from information_schema.views
        where table_schema not in ('information_schema', 'pg_catalog')
        order by schema_name,
                view_name;
        """)

    def show_columns(self, schema, table):
        self.read("""
        SELECT * 
        FROM information_schema.columns
        WHERE 
            table_schema = 'myschema' 
            AND table_name = 'mytable';
        """)

    def show_tables(self, schema=None):
        if schema:
            return self.read(f"""
            SELECT 
                CAST(table_catalog AS VARCHAR(30)) AS table_catalog, 
                table_schema, table_name, 
                CAST(table_type AS VARCHAR(30)) AS table_type
            FROM information_schema.tables
            WHERE table_schema = '{schema}'
            UINION ALL
            select 
                CAST('external' AS VARCHAR(30)) as table_catalog,
                schemaname as table_schema, 
                tablename as table_name,
                CAST('SPECTRUM' AS VARCHAR(30)) as table_type
            from svv_external_tables
            """
            )
        else:
            return self.read(f"""
            SELECT 
                CAST(table_catalog AS VARCHAR(30)) AS table_catalog, 
                table_schema, table_name, 
                CAST(table_type AS VARCHAR(30)) AS table_type
            FROM information_schema.tables
            WHERE table_schema not in ('pg_catalog', 'information_schema')
            UNION ALL
            select 
                CAST('external' AS VARCHAR(30)) as table_catalog,
                schemaname as table_schema, 
                tablename as table_name,
                CAST('SPECTRUM' AS VARCHAR(30)) as table_type
            from svv_external_tables
            """)

    def show_schemas(self, owner='datalakeadmin'):
        return self.read(f"""
            select s.nspname as table_schema,
                s.oid as schema_id,  
                u.usename as owner
            from pg_catalog.pg_namespace s
            join pg_catalog.pg_user u on u.usesysid = s.nspowner
            where owner = '{owner}'
            order by table_schema;
            """)

    def read_table(self, table_name, limit=None):
        
        if self._conn_type == 'aws_redshift':
            limit = f"LIMIT {limit}" if limit else ''
            query = "SELECT * FROM {} {}".format(table_name, limit)
        else:
            limit = f"TOP {limit}" if limit else ''
            query = "SELECT {} * FROM {}".format(limit, table_name)
        self._df = read_sql(query, self.conn)
        self._last_columns = self._df.columns.tolist()
        return self._df

    def get_columns(self, table=None, find=None):

        if self._last_columns == None and table == None:
            if self._conn_type == 'mssql':
                query = "SELECT TOP 1 * FROM {}".format(table)
            else:
                query = "SELECT * FROM {} LIMIT 1".format(table)
            df = read_sql(query, self.conn)
            columns = df.columns.tolist()
        else:
            columns = self._last_columns

        if find:
            if isinstance(find, str): find = [find]
            func = lambda col: \
                any([f.lower() in col.lower() for f in find])
            columns = list(filter(func , columns))

        return columns

    def generate_create(self, schema, name, 
        outer_dataframe=None, columns=None, dtypes=None, primary_keys=None):

        # creating the create stement structure
        create = """CREATE TABLE {}.{} (\n{}{}\n)"""
        if type(outer_dataframe) != None:
            columns = outer_dataframe.columns.to_list()

        # no columns error handling
        if self._last_columns == None and columns == None:
            raise ValueError("No inplicit columns and no given columns. "
                            "Try providing columns or running a quary first.")

        # establishing columns
        columns = columns or self._last_columns        
        dtypes = dtypes or ['VARCHAR NULL'] * len(columns)
        zip_ = zip(dtypes , columns)
        func = lambda x: f"\t{x[1]} {x[0]}"
        statement = ',\n'.join(list(map(func, zip_)))

        if primary_keys:
            primary_keys = f",\n\tPRIMARY KEY ({', '.join(primary_keys)})"
        primary_keys =  primary_keys or ''

        # constructing create statement
        create_statement = create.format(schema, name, statement, primary_keys)

        return create_statement
            
    
    def execute(self, query, commit=True, ignore_error=False):
        conn = self.conn
        cursor = self.cursor
        
        try:
            cursor.execute(query)
            print("Query executed successfully!")
        except Exception as e:
            # on error rollback transaction
            conn.rollback()
            print("Rolled back.")
            if not ignore_error:
                raise RuntimeError(e, "The transaction was rolled back")

        if commit: conn.commit()

    def execute_file(self, filepath, show=True):
        with open(filepath) as sql_file:
            sql_query = sql_file.read()
            if show: print(sql_query)
            self.execute(sql_query)

    def _validating_secret_user_cred(self, root_path, file_path):
        path_exist = os.path.exists(root_path)
        file_exist = os.path.exists(file_path)

        if not path_exist:
            os.mkdir(root_path)
        
        if not file_exist:
            with open(file_path, 'w+') as f:
                f.write('[aws_redshift]\nuser=\npassword='
                    '\n[mssql_server]\nuser=\npassword=')
                f.close()

    def _validating_config_folder(self):
        root_path = self.root_path
        root_exist = os.path.exists(root_path)

        if not root_exist:
            # creates the .config folder
            os.mkdir(root_path)

    def _validating_config_file(self):
        # extracting attributes
        full_path = self.full_path

        # checking if file exist
        full_path_exist = os.path.exists(full_path)

        if not full_path_exist:
            # create the file
            with open(full_path, 'w+') as f:
                with open('default-ini.txt', 'r+') as default_file:
                    default_ini = default_file.read()
                f.write(default_ini)

                f.close()

    def add_server(self, conn_type, server_endpoint, server_port):
        # extracting attributes

        # setting server drivers depending on connection
        pass

        # constructing server information dictionary
        # new_server_dict = dict(
        #     endpoint=server_endpoint,
        #     driver=server_driver,
        #     port=server_port)
