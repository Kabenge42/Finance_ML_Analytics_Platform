-- ============================================================================
-- Equities Table Access Control and Permissions
-- ============================================================================
-- Description: This script implements role-based access control (RBAC) for
--              the equities table following the principle of least privilege.
--
-- Security Best Practices Applied:
--   - Role-based access control instead of direct user grants
--   - Minimum necessary privileges per role
--   - Schema-qualified table names
--   - Transaction-based grants for atomicity
--   - Comprehensive documentation
--
-- Roles Created:
--   - equities_readonly: Read-only access for reporting/analytics
--   - equities_app: CRUD operations for application users
--   - equities_admin: Full administrative access (use sparingly)
--
-- Usage:
--   1. Execute this script as a database superuser
--   2. Assign roles to actual users: GRANT role_name TO username;
--   3. Regularly audit role assignments and permissions
--
-- Date Created: 2025-10-26
-- ============================================================================

-- Start transaction for atomic execution
BEGIN;

-- ============================================================================
-- STEP 1: Create Roles
-- ============================================================================

-- Read-only role for reporting, analytics, and data science users
DO
$$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'equities_readonly') THEN
            CREATE ROLE equities_readonly
                NOLOGIN
                NOSUPERUSER
                NOCREATEDB
                NOCREATEROLE
                NOINHERIT;
        END IF;
    END
$$;

COMMENT ON ROLE equities_readonly IS
    'Read-only access to equities table for reporting and analytics';

-- Application role for CRUD operations
DO
$$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'equities_app') THEN
            CREATE ROLE equities_app
                NOLOGIN
                NOSUPERUSER
                NOCREATEDB
                NOCREATEROLE
                NOINHERIT;
        END IF;
    END
$$;

COMMENT ON ROLE equities_app IS
    'Application-level access with CRUD permissions on equities table';

-- Administrative role for DDL and maintenance operations
DO
$$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'equities_admin') THEN
            CREATE ROLE equities_admin
                NOLOGIN
                NOSUPERUSER
                NOCREATEDB
                NOCREATEROLE
                NOINHERIT;
        END IF;
    END
$$;

COMMENT ON ROLE equities_admin IS
    'Full administrative access to equities table - assign with caution';

-- ============================================================================
-- STEP 2: Grant Permissions to Roles
-- ============================================================================

-- Read-only role: SELECT only
GRANT SELECT ON public.equities TO equities_readonly;

-- Application role: Standard CRUD operations
GRANT SELECT, INSERT, UPDATE, DELETE ON public.equities TO equities_app;

-- Admin role: Full privileges including DDL and dangerous operations
-- Note: TRUNCATE and TRIGGER are powerful operations - use with extreme caution
GRANT ALL PRIVILEGES ON public.equities TO equities_admin;

-- Grant usage on schema if needed
GRANT USAGE ON SCHEMA public TO equities_readonly;
GRANT USAGE ON SCHEMA public TO equities_app;
GRANT USAGE ON SCHEMA public TO equities_admin;

-- ============================================================================
-- STEP 3: Grant Sequence Access (if applicable)
-- ============================================================================
-- If the equities table uses a sequence for auto-incrementing IDs,
-- grant USAGE and SELECT on the sequence to the app role

-- Uncomment and adjust the sequence name if needed:
-- GRANT USAGE, SELECT ON SEQUENCE public.equities_id_seq TO equities_app;

-- ============================================================================
-- STEP 4: Table Documentation
-- ============================================================================

COMMENT ON TABLE public.equities IS
    'Equity securities data with role-based access control. Roles: equities_readonly (SELECT), equities_app (CRUD), equities_admin (ALL)';

-- ============================================================================
-- STEP 5: Assign Roles to Users (Examples - Customize as needed)
-- ============================================================================

-- IMPORTANT: Replace these example usernames with your actual database users
-- Uncomment and modify the lines below according to your environment:

-- Example: Grant read-only access to reporting users
-- GRANT equities_readonly TO reporting_user;
-- GRANT equities_readonly TO data_analyst;
-- GRANT equities_readonly TO bi_service_account;

-- Example: Grant application access to service accounts
-- GRANT equities_app TO application_service_account;
-- GRANT equities_app TO etl_service_account;

-- Example: Grant admin access to database administrators (use sparingly)
-- GRANT equities_admin TO database_admin;
-- GRANT equities_admin TO senior_dba;

-- ============================================================================
-- STEP 6: Verify Permissions (Optional - for testing)
-- ============================================================================

-- Query to check granted permissions:
-- SELECT 
--     grantee,
--     table_schema,
--     table_name,
--     privilege_type
-- FROM 
--     information_schema.table_privileges
-- WHERE 
--     table_name = 'equities'
-- ORDER BY 
--     grantee, privilege_type;

-- ============================================================================
-- STEP 7: Revoke Public Access (Security Hardening)
-- ============================================================================

-- By default, PostgreSQL may grant some permissions to PUBLIC
-- Revoke these to ensure only explicit role assignments have access
REVOKE ALL ON public.equities FROM PUBLIC;

-- Commit the transaction
COMMIT;

-- ============================================================================
-- STEP 8: Set Default Privileges (Optional but Recommended)
-- ============================================================================

-- Ensure future tables/sequences in the schema inherit these permissions
-- Replace 'your_schema_owner' with the actual schema owner role

-- ALTER DEFAULT PRIVILEGES FOR ROLE your_schema_owner IN SCHEMA public
--     GRANT SELECT ON TABLES TO equities_readonly;

-- ALTER DEFAULT PRIVILEGES FOR ROLE your_schema_owner IN SCHEMA public
--     GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO equities_app;

-- ALTER DEFAULT PRIVILEGES FOR ROLE your_schema_owner IN SCHEMA public
--     GRANT ALL PRIVILEGES ON TABLES TO equities_admin;

-- Commit the transaction
COMMIT;

-- ============================================================================
-- MAINTENANCE NOTES
-- ============================================================================
--
-- To revoke a role from a user:
--   REVOKE role_name FROM username;
--
-- To revoke permissions from a role:
--   REVOKE privilege_type ON public.equities FROM role_name;
--
-- To drop a role (must revoke from all users first):
--   DROP ROLE IF EXISTS role_name;
--
-- To audit current permissions:
--   \dp equities  (in psql)
--   or use the query in STEP 6 above
--
-- Security Checklist:
--   □ Regularly review role assignments
--   □ Remove access for departing team members
--   □ Audit permission changes in production
--   □ Use least privilege principle
--   □ Never grant SUPERUSER for application access
--   □ Limit TRUNCATE and TRIGGER permissions
--   □ Monitor for unauthorized access attempts
--
-- ============================================================================

-- End of script
