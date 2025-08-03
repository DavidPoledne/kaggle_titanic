import pandas as pd
import os

folder_path = "/home/david/Documents/data_science_1/kaggle/titanic/model_comparison/analysis/data_create_one_xlsx"
output_file = "model_comparison.xlsx"

# Create a Pandas ExcelWriter object
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            sheet_name = os.path.splitext(filename)[0][:31]  # Excel allows max 31 characters in sheet name
            df = pd.read_csv(file_path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Combined file saved as {output_file}")