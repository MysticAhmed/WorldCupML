"""
Extract the dataset from the provided document and integrate it.
This script processes the complete FIFA World Cup 1930-2022 dataset.
"""

import pandas as pd
import os

def extract_dataset_from_document():
    """
    The document contains 964 World Cup matches from 1930-2022.
    Since the file already exists with the complete data from the document,
    we'll verify it and proceed with integration.
    """
    
    dataset_file = "FIFA World Cup 1930-2022 All Match Dataset.csv"
    
    if not os.path.exists(dataset_file):
        print(f"ERROR: {dataset_file} not found!")
        print("\nThe document you provided contains 964 rows of World Cup data.")
        print("Please save the complete CSV content from the document to this file.")
        return False
    
    # Read and verify the dataset
    try:
        df = pd.read_csv(dataset_file)
        print(f"✓ Dataset loaded: {len(df)} matches")
        
        # Parse dates
        df['Match Date'] = pd.to_datetime(df['Match Date'], format='%m/%d/%Y', errors='coerce')
        df['Year'] = df['Match Date'].dt.year
        
        # Check for 2018 and 2022 data
        years = sorted(df['Year'].dropna().unique())
        print(f"✓ Years covered: {min(years)} to {max(years)}")
        
        matches_2018 = len(df[df['Year'] == 2018])
        matches_2022 = len(df[df['Year'] == 2022])
        
        print(f"✓ 2018 World Cup: {matches_2018} matches")
        print(f"✓ 2022 World Cup: {matches_2022} matches")
        
        if matches_2018 == 0 or matches_2022 == 0:
            print("\n⚠ WARNING: Missing 2018 or 2022 data!")
            print("The document should contain 64 matches for each tournament.")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR reading dataset: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("FIFA World Cup Dataset Extraction & Integration")
    print("=" * 70)
    print()
    
    if extract_dataset_from_document():
        print("\n" + "=" * 70)
        print("Dataset verified! Now running integration...")
        print("=" * 70)
        print()
        
        # Import and run the integration
        from integrate_new_data import integrate_new_data
        integrate_new_data()
    else:
        print("\n" + "=" * 70)
        print("Please ensure the complete dataset is saved first.")
        print("=" * 70)
