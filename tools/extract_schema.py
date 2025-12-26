import sys

# Add the project root to sys.path to import existing schema
sys.path.append('.')

try:
    from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA as OLD_SCHEMA
    import json
    
    with open('schema_dump.json', 'w') as f:
        json.dump(OLD_SCHEMA, f, indent=2)
    print('SUCCESS')
except Exception as e:
    print(f'Error: {e}')
