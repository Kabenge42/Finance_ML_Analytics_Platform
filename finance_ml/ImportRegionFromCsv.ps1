# Database connection parameters
$DbConfig = [PSCustomObject]@{
    Host = "localhost"
    Port = "5432"
    User = "postgres"
    Name = "postgres"
}

# Function to build psql connection arguments
function Get-PsqlConnectionArgs
{
    param([PSCustomObject]$Config)
    return @("-h", $Config.Host, "-p", $Config.Port, "-U", $Config.User, "-d", $Config.Name)
}

# Function to execute psql command with connection parameters
function Invoke-PsqlCommand
{
    param(
        [PSCustomObject]$DbConfig,
        [string]$Command
    )

    $connectionArgs = Get-PsqlConnectionArgs -Config $DbConfig
    $allArgs = $connectionArgs + @("-c", $Command)

    try
    {
        psql @allArgs
        if ($LASTEXITCODE -ne 0)
        {
            throw "psql command failed with exit code $LASTEXITCODE"
        }
    }
    catch
    {
        Write-Error "Database command failed: $_"
        throw
    }
}

# Function to import a region
function Import-Region
{
    param(
        [string]$CsvFile,
        [string]$RegionCode,
        [PSCustomObject]$DbConfig
    )

    Write-Host "Importing $RegionCode from $CsvFile..."

    try
    {
        # Create staging table
        Invoke-PsqlCommand -DbConfig $DbConfig -Command "CREATE TEMP TABLE equities_staging (LIKE equities)"

        # Copy data from CSV
        $copyCommand = "\copy equities_staging FROM '$CsvFile' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')"
        Invoke-PsqlCommand -DbConfig $DbConfig -Command $copyCommand

        # Update region
        $updateCommand = "UPDATE equities_staging SET `"Region`"='$RegionCode' WHERE `"Region`" IS NULL"
        Invoke-PsqlCommand -DbConfig $DbConfig -Command $updateCommand

        # Insert into main table
        Invoke-PsqlCommand -DbConfig $DbConfig -Command "INSERT INTO equities SELECT * FROM equities_staging ON CONFLICT DO NOTHING"

        Write-Host "✓ $RegionCode imported successfully"
    }
    catch
    {
        Write-Error "Failed to import region $RegionCode : $_"
        throw
    }
}

# Import all regions
$regions = @(
    @{ File = "data\screening_us.csv"; Code = "US" },
    @{ File = "data\screening_eu.csv"; Code = "EU" },
    @{ File = "data\screening_apac.csv"; Code = "APAC" },
    @{ File = "data\screening_rotw.csv"; Code = "ROTW" }
)

foreach ($region in $regions)
{
    Import-Region -CsvFile $region.File -RegionCode $region.Code -DbConfig $DbConfig
}

Write-Host "`n✓ All regions imported successfully"
