import pandas as pd

def process_csv():
    try:
        # Read the keypoint.csv file
        df = pd.read_csv('keypoint.csv')
        
        # Convert first column to string to ensure consistent processing
        first_col = df.columns[0]
        df[first_col] = df[first_col].astype(str)
        
        # Delete rows where first column starts with 4
        df = df[~df[first_col].str.startswith('4')]
        
        # Replace 5 with 4 at the start of values
        df[first_col] = df[first_col].apply(
            lambda x: '4' + x[1:] if x.startswith('5') else x
        )
        
        # Convert first column to numeric for sorting
        # Extract first number from each value for sorting
        df['sort_key'] = df[first_col].str.extract('(\d+)').astype(int)
        
        # Sort by the extracted number
        df = df.sort_values('sort_key')
        
        # Remove the temporary sorting column
        df = df.drop('sort_key', axis=1)
        
        # Save the processed data to output.csv
        df.to_csv('output.csv', index=False)
        print("File processed successfully. Output saved to output.csv")
        
    except FileNotFoundError:
        print("Error: keypoint.csv not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    process_csv()