import json

with open('ml_finance_model_main_v10.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

with open('sector_models_output.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(cells):
        source = ''.join(c['source'])
        if 'sector_models_result' in source:
            out.write(f'CELL {i}:\n')
            out.write(source)
            out.write('\n' + '---' * 20 + '\n\n')

print("Output written to sector_models_output.txt")
