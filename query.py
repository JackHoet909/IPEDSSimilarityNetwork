import pandas as pd
import argparse

#Create a parser for command line arguments
parser = argparse.ArgumentParser(description="Removing rows or removing columns based on certain criteria")
parser.add_argument("--CSVfile", help="The name of the CSV file to read in")
parser.add_argument("--XLSXFile", help="The name of the Excel file to read in")
parser.add_argument("--query", help="The query to filter the data by")
parser.add_argument("--output", help="The name of the output CSV file")
args = parser.parse_args()

if args.XLSXFile:
    df = pd.read_excel(args.XLSXFile, index_col=0)
else:
    df = pd.read_csv(args.CSVfile, encoding="latin1", index_col=0)
if not args.CSVfile and not args.XLSXFile:
    print("Please provide a CSV or XLSX file to read in using the --CSVfile or --XLSXFile argument.")
    exit(1)
df = df.query(args.query)
print(df)

df.to_csv(args.output)