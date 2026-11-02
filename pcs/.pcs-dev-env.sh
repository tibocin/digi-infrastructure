# PCS lightweight dev environment (SQLite, dev secrets). Source before running the server.
# Idempotent: only exports env vars. Safe to re-source. Contains DEV-ONLY placeholder secrets.
export PATH="$HOME/.local/bin:$PATH"
export PCS_ENVIRONMENT=development
export PCS_DEBUG=true
export PCS_DB_DIALECT=sqlite+aiosqlite
export PCS_DB_NAME=pcs_dev.db
export PCS_SECURITY_SECRET_KEY=dev_secret_key_change_me_0123456789
export PCS_SECURITY_JWT_SECRET_KEY=dev_jwt_secret_key_change_me_0123456789
