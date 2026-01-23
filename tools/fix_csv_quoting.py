import csv


def fix_csv_quoting(input_path, output_path):
    """Fix CSV files with improperly escaped quotes in Description field."""
    with open(input_path, "r", encoding="utf-8") as infile:
        content = infile.read()

    # Replace internal double quotes that aren't properly escaped
    # This regex finds quoted fields and escapes internal quotes
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(content.splitlines())
        writer = csv.writer(
            outfile, quoting=csv.QUOTE_ALL, escapechar="\\", doublequote=True
        )
        for row in reader:
            writer.writerow(row)


# Run for each file
for region in ["us", "eu", "apac", "rotw"]:
    fix_csv_quoting(
        f"data/screening_{region}.csv", f"data/screening_{region}_fixed.csv"
    )
