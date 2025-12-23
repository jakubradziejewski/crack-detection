import pandas as pd

def process_final_submission(input_path, output_path):
    df = pd.read_csv(input_path)
    df.columns = ['file', 'crack']

    df['file'] = df['file'].astype(str)
    
    df['file'] = df['file'].str.replace('.jpg', '', regex=False)
    
    df['file'] = df['file'] + '.jpg'
    
    mask = df['file'].str.startswith('noncrack_noncrack')
    df.loc[mask, 'file'] = df.loc[mask, 'file'] + '.jpg'
    
    df.to_csv(output_path, index=False)
    
    total = len(df)
    noncrack_count = mask.sum()
    print(f"Correctly transformed {total} lines.")

process_final_submission('submission.csv', 'submission_final.csv')