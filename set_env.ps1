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
$env:MODEL_DIR = "models"
$env:CACHE_DIR = ".cache"
$env:OUTPUT_DIR = "outputs"

# Model configuration
$env:MODEL_VERSION = "v8_3"
$env:RANDOM_SEED = "42"

# Performance settings
$env:N_JOBS = "4"

# Logging configuration
$env:LOG_LEVEL = "INFO"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

# Database configuration (uncomment and configure if needed)
# $env:DB_URL = "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
# $env:DB_SCHEMA = "public"
# $env:DB_TABLE = "equities"

# API Keys (uncomment and configure if needed)
# $env:ALPHA_VANTAGE_API_KEY = "your_api_key_here"
# $env:FINANCIAL_API_KEY = "your_api_key_here"

Write-Host "Environment variables set successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Active Configuration:" -ForegroundColor Cyan
Write-Host "  DATA_DIR:              $env:DATA_DIR"
Write-Host "  MODEL_DIR:             $env:MODEL_DIR"
Write-Host "  OUTPUT_DIR:            $env:OUTPUT_DIR"
Write-Host "  MODEL_VERSION:         $env:MODEL_VERSION"
Write-Host "  RANDOM_SEED:           $env:RANDOM_SEED"
Write-Host "  N_JOBS:                $env:N_JOBS"
Write-Host "  TF_CPP_MIN_LOG_LEVEL:  $env:TF_CPP_MIN_LOG_LEVEL"
