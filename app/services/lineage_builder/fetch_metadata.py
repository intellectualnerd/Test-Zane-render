import pandas as pd
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL
import os
from dotenv import load_dotenv

def transfer_snowflake_to_postgres(
    snowflake_db_filter='PROD_TZ',
    snowflake_schema_list=('DEMOGRAPHIC','PROPERTY','AGILE_REPORTING','CONSERVICE','CUSTOMER_ACTIVITY','DATADOG','EDW','EDW_SNF_DBO','EDW_STAGING','HR_MANAGEMENT','IMG_CAPITAL_ALLOCATION','IMG_DISPOSITION_TRACKER','INVH_SALESFORCE','LEARNING_AND_DEVELOPMENT','LEARNUPON','MLS','NATIONAL_RTM','OCTANTIS','OTM','PLAID','POLARIS','REV_MGMT','SMART_RENT','SYNDICATION','TABLEAU_OUTPUT_CONSOLIDATION','TASKEASY','USER_EXPERIENCE_SURVEY','WEB_RATING'),
    days_back=1,
    pg_query_access_history_table_name='snowflake_query_history',
    pg_information_schema_columns_table_name='information_schema_columns'
):
    # Load environment variables from .env
    load_dotenv()

    snowflake_engine = None
    pg_engine = None

    try:
        # Snowflake config
        snowflake_config = {
            'user': os.getenv('SNOWFLAKE_USER'),
            # 'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
            'role': os.getenv('SNOWFLAKE_ROLE')
        }

        # Create Snowflake engine
        snowflake_engine = create_engine(URL(
            user=snowflake_config['user'],
            # password=snowflake_config['password'],
            account=snowflake_config['account'],
            warehouse=snowflake_config['warehouse'],
            role=snowflake_config['role'],
            authenticator='externalbrowser'
        ))

        # Build SQL queries
        query_access_history_query = f"""
                WITH ranked_queries AS (
                        SELECT query_text,
                            query_id,
                            query_type,
                            start_time,
                            end_time,
                            database_id,
                            database_name,
                            schema_id,
                            schema_name,
                            session_id,
                            ROW_NUMBER() OVER (
                                PARTITION BY query_text
                                ORDER BY start_time DESC
                            ) AS rn
                        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
                        WHERE DATABASE_NAME = '{snowflake_db_filter}'
                            AND SCHEMA_NAME IN ({','.join(f"'{s}'" for s in snowflake_schema_list)})
                            AND START_TIME >= DATEADD(DAY, -{days_back}, CURRENT_TIMESTAMP())
                            AND QUERY_TYPE IN ('INSERT', 'MERGE', 'CREATE_VIEW') AND execution_status = 'SUCCESS'
                    )
                    SELECT 
                        '76d33fb3-6062-456b-a211-4aec9971f8be' AS org_id,
                        '32f55d8f-4731-4810-aeb8-4cec0d5ae989' AS batch_id,
                        '4aeb318b-6819-4873-9fae-33bab55ac922' AS connection_id,
                        ah.query_id,
                        rq.query_text,
                        rq.database_name,
                        rq.database_id,
                        rq.schema_name,
                        rq.schema_id,
                        rq.query_type,
                        rq.start_time,
                        rq.end_time,
                        rq.session_id,
                        ah.base_objects_accessed,
                        ah.objects_modified
                    FROM SNOWFLAKE.ACCOUNT_USAGE.ACCESS_HISTORY ah
                    JOIN ranked_queries rq
                    ON ah.query_id = rq.query_id
                    WHERE rq.rn = 1 
                    ORDER BY rq.start_time DESC;
                    """

        
        query_information_schema_columns = f"""
            SELECT '76d33fb3-6062-456b-a211-4aec9971f8be' AS org_id,
                   '4aeb318b-6819-4873-9fae-33bab55ac922' AS connection_id,
                    TABLE_CATALOG as table_catalog,
                    TABLE_SCHEMA as table_schema,
                    TABLE_NAME as table_name,
                    COLUMN_NAME as column_name,
                    DATA_TYPE as data_type,
                    ORDINAL_POSITION as ordinal_position
            FROM SNOWFLAKE.ACCOUNT_USAGE.COLUMNS
            WHERE DELETED IS NULL;
        """

        # Fetch data from Snowflake
        query_access_history_df = pd.read_sql(text(query_access_history_query), snowflake_engine)
        print(f"Retrieved {len(query_access_history_df)} rows from Snowflake ACCESS_HISTORY.")

        information_schema_columns_df = pd.read_sql(text(query_information_schema_columns), snowflake_engine)
        print(f"Retrieved {len(information_schema_columns_df)} rows from Snowflake INFORMATION_SCHEMA.COLUMNS.")

        # Connect to PostgreSQL
        pg_engine = create_engine(os.getenv('DATABASE_URL'))

        # Write to PostgreSQL
        query_access_history_df.to_sql(pg_query_access_history_table_name, pg_engine, if_exists='append', index=False)
        print(f"Inserted {len(query_access_history_df)} rows into PostgreSQL table '{pg_query_access_history_table_name}'.")

        information_schema_columns_df.to_sql(pg_information_schema_columns_table_name, pg_engine, if_exists='append', index=False)
        print(f"Inserted {len(information_schema_columns_df)} rows into PostgreSQL table '{pg_information_schema_columns_table_name}'.")

    except Exception as e:
        print(f"Error during transfer: {e}")

    finally:
        if snowflake_engine:
            snowflake_engine.dispose()
            print("Snowflake engine disposed.")
        if pg_engine:
            pg_engine.dispose()
            print("PostgreSQL engine disposed.")


if __name__ == '__main__':
    transfer_snowflake_to_postgres()
