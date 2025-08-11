#!/bin/bash
# -----------------------------------------------------------------------------
# File: scripts/init-multiple-databases.sh
# Purpose: Initialize multiple databases for different apps in PostgreSQL
# Related: docker-compose.yml, config/env.example, pcs/.env
# Tags: database, postgres, multi-app, security
# -----------------------------------------------------------------------------

set -e

# Function to create database and user
create_database() {
    local database=$1
    local user=$2
    local password=$3
    
    echo "Creating database '$database' for app '$user' with password: ${password:0:3}***"
    
    # Create database
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        CREATE DATABASE $database;
        CREATE USER $user WITH PASSWORD '$password';
        GRANT ALL PRIVILEGES ON DATABASE $database TO $user;
        GRANT CREATE ON DATABASE $database TO $user;
EOSQL
    
    echo "Database '$database' created successfully"
}

# Function to get password from environment variable
get_password() {
    local database=$1
    local env_var="${database^^}_DB_PASSWORD"  # Convert to uppercase
    local default_password="${database}_pass"   # Fallback default
    
    # Try to get password from environment variable
    if [ -n "${!env_var}" ]; then
        echo "${!env_var}"
        echo "Using password from environment variable: $env_var" >&2
    else
        echo "$default_password"
        echo "Using default password: $default_password (set $env_var to override)" >&2
    fi
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
        password=$(get_password "$database")
        
        # Create database and user
        create_database "$database" "$user" "$password"
    done
    
    echo "All databases initialized successfully"
else
    echo "No additional databases specified, using default setup"
fi


