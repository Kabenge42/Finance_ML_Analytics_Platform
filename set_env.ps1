# =============================================================================
# Finance ML Analytics Platform - Environment Variables Setup
# =============================================================================
# PowerShell script to set environment variables for the current session
#
# Usage:
#   .\set_env.ps1
#   or
#   . .\set_env.ps1  (dot-source to persist in current session)
#
# To make permanent (user-level):
#   [Environment]::SetEnvironmentVariable("VAR_NAME", "value", "User")
# =============================================================================

# Disable colored output
$env:NO_COLOR = "1"

# Directory paths
$env:DATA_DIR = "data"
$env:MODEL_DIR = "regression"
$env:CACHE_DIR = ".cache"
$env:OUTPUT_DIR = "outputs"

# Model configuration
$env:MODEL_VERSION = "v9_9"
$env:RANDOM_SEED = "42"

# Performance settings
$env:N_JOBS = "4"

# Logging configuration
$env:LOG_LEVEL = "INFO"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

# Database connection (update with your actual credentials)
$env:DB_URL = "postgresql+psycopg2://postgres:bItcfiTg142!@localhost:5432/postgres"

Write-Host "Environment variables set successfully!" -ForegroundColor Green
Write-Host "NO_COLOR: $env:NO_COLOR"
Write-Host "DATA_DIR: $env:DATA_DIR"
Write-Host "OUTPUT_DIR: $env:OUTPUT_DIR"
Write-Host "MODEL_DIR: $env:MODEL_DIR"
Write-Host "RANDOM_SEED: $env:RANDOM_SEED"
