"""
Filepath: pcs/scripts/db.py
Purpose: Database management script for migrations and operations
Related Components: Database, Alembic, migrations
Tags: database, migrations, alembic, management
"""

#!/usr/bin/env python3
"""
Database management script for PCS system.
Provides commands for database operations, migrations, and maintenance.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def run_command(command: str, cwd: Path = None) -> int:
    """Run a shell command and return exit code."""
    if cwd is None:
        cwd = project_root
    
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    return result.returncode

def check_database_connection() -> bool:
    """Check if database connection is available."""
    try:
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get database URL from environment
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL environment variable not set")
            return False
        
        # Try to create a simple connection
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def create_database() -> int:
    """Create the database if it doesn't exist."""
    print("Creating database...")
    
    # This would typically use psql or similar
    # For now, we'll just check the connection
    if check_database_connection():
        print("Database is accessible")
        return 0
    else:
        print("Please ensure PostgreSQL is running and accessible")
        return 1

def run_migrations() -> int:
    """Run database migrations."""
    print("Running database migrations...")
    return run_command("alembic upgrade head")

def create_migration(message: str) -> int:
    """Create a new migration."""
    print(f"Creating migration: {message}")
    return run_command(f'alembic revision -m "{message}"')

def rollback_migration(revision: str) -> int:
    """Rollback to a specific migration revision."""
    print(f"Rolling back to revision: {revision}")
    return run_command(f"alembic downgrade {revision}")

def show_migration_history() -> int:
    """Show migration history."""
    print("Migration history:")
    return run_command("alembic history")

def show_current_revision() -> int:
    """Show current migration revision."""
    print("Current migration revision:")
    return run_command("alembic current")

def reset_database() -> int:
    """Reset database to initial state."""
    print("Resetting database...")
    
    # Drop all tables
    result = run_command("alembic downgrade base")
    if result != 0:
        return result
    
    # Run migrations to recreate tables
    return run_command("alembic upgrade head")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PCS Database Management")
    parser.add_argument('command', choices=[
        'check', 'create', 'migrate', 'create-migration', 'rollback', 
        'history', 'current', 'reset'
    ], help='Command to execute')
    parser.add_argument('--message', '-m', help='Migration message (for create-migration)')
    parser.add_argument('--revision', '-r', help='Revision for rollback')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        return 0 if check_database_connection() else 1
    elif args.command == 'create':
        return create_database()
    elif args.command == 'migrate':
        return run_migrations()
    elif args.command == 'create-migration':
        if not args.message:
            print("Error: Migration message is required")
            return 1
        return create_migration(args.message)
    elif args.command == 'rollback':
        if not args.revision:
            print("Error: Revision is required")
            return 1
        return rollback_migration(args.revision)
    elif args.command == 'history':
        return show_migration_history()
    elif args.command == 'current':
        return show_current_revision()
    elif args.command == 'reset':
        return reset_database()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
