#!/bin/bash
# -----------------------------------------------------------------------------
# File: scripts/init-multiple-databases.sh
# Purpose: Initialize multiple databases for different apps in PostgreSQL
# Related: docker-compose.yml, config/env.example
# Tags: database, postgres, multi-app
# -----------------------------------------------------------------------------

set -e

# Function to create database and user
create_database() {
    local database=$1
    local user=$2
    local password=$3
    
    echo "Creating database '$database' for app '$user'"
    
    # Create database
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        CREATE DATABASE $database;
        CREATE USER $user WITH PASSWORD '$password';
        GRANT ALL PRIVILEGES ON DATABASE $database TO $user;
        GRANT CREATE ON DATABASE $database TO $user;
EOSQL
    
    echo "Database '$database' created successfully"
}

# Get list of databases from environment variable
if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Initializing multiple databases: $POSTGRES_MULTIPLE_DATABASES"
    
    # Split comma-separated list
    IFS=',' read -ra DATABASES <<< "$POSTGRES_MULTIPLE_DATABASES"
    
    for database in "${DATABASES[@]}"; do
        # Trim whitespace
        database=$(echo "$database" | xargs)
        
        # Create user and password (using database name as base)
        user="${database}_user"
        password="${database}_pass"
        
        # Create database and user
        create_database "$database" "$user" "$password"
    done
    
    echo "All databases initialized successfully"
else
    echo "No additional databases specified, using default setup"
fi


