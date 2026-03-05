-- Initialize the database with proper permissions
-- This script runs automatically when the PostgreSQL container starts for the first time

-- The database 'queryguard' is automatically created by the PostgreSQL container
-- because we set POSTGRES_DB=queryguard in the environment variables.
-- This script runs in the context of that database.

-- Create any additional extensions if needed
-- Uncomment these if your application requires them:
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Grant necessary permissions to the user
-- (The user already has privileges, but ensuring they have all needed permissions)
GRANT ALL PRIVILEGES ON DATABASE queryguard TO queryguard_user;

-- Grant schema permissions for future tables
GRANT ALL ON SCHEMA public TO queryguard_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO queryguard_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO queryguard_user;

-- Grant default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO queryguard_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO queryguard_user;

-- The application will create tables automatically using SQLAlchemy
-- No need to create tables manually as the app handles this via Base.metadata.create_all()