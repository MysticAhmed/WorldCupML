"""
Integrate the new FIFA World Cup 1930-2022 dataset into the existing project.
This script will update the WorldCupMatches.csv file with data from 2018 and 2022.
"""

import pandas as pd
import os

def integrate_new_data():
    """
    Read the existing WorldCupMatches.csv and add/update with newer data.
    The new dataset covers 1930-2022, so we'll extract 2018 and 2022 data.
    """
    
    # Check if the new dataset file exists
    new_dataset_file = "FIFA World Cup 1930-2022 All Match Dataset.csv"
    
    if not os.path.exists(new_dataset_file):
        print(f"Error: {new_dataset_file} not found!")
        print("\nPlease create this file with the dataset content from the document.")
        print("You can copy the CSV data from the document you provided.")
        return False
    
    print(f"Reading new dataset from {new_dataset_file}...")
    # Try different encodings to handle special characters
    try:
        new_df = pd.read_csv(new_dataset_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            new_df = pd.read_csv(new_dataset_file, encoding='latin-1')
        except UnicodeDecodeError:
            new_df = pd.read_csv(new_dataset_file, encoding='cp1252')
    
    # Filter for 2018 and 2022 World Cups only (the new data)
    new_df['Match Date'] = pd.to_datetime(new_df['Match Date'], format='%m/%d/%Y', errors='coerce')
    new_df['Year'] = new_df['Match Date'].dt.year
    
    recent_data = new_df[new_df['Year'].isin([2018, 2022])].copy()
    
    if len(recent_data) == 0:
        print("No 2018 or 2022 data found in the new dataset!")
        return False
    
    print(f"Found {len(recent_data)} matches from 2018 and 2022")
    print(f"  2018: {len(recent_data[recent_data['Year'] == 2018])} matches")
    print(f"  2022: {len(recent_data[recent_data['Year'] == 2022])} matches")
    
    # Convert to the existing format
    converted = pd.DataFrame()
    
    converted['Year'] = recent_data['Year']
    converted['Datetime'] = recent_data['Match Date'].dt.strftime('%d %b %Y') + ' - ' + recent_data['Match Time'].fillna('00:00')
    
    # Map stage names
    stage_mapping = {
        'group stage': 'Group Stage',
        'round of 16': 'Round of 16',
        'quarter-finals': 'Quarter-finals',
        'semi-finals': 'Semi-finals',
        'final': 'Final',
        'third-place match': 'Third Place',
    }
    converted['Stage'] = recent_data['Stage Name'].str.lower().map(stage_mapping).fillna(recent_data['Stage Name'])
    
    converted['Stadium'] = recent_data['Stadium Name']
    converted['City'] = recent_data['City Name']
    converted['Home Team Name'] = recent_data['Home Team Name']
    converted['Home Team Goals'] = recent_data['Home Team Score'].astype(float)
    converted['Away Team Goals'] = recent_data['Away Team Score'].astype(float)
    converted['Away Team Name'] = recent_data['Away Team Name']
    
    # Win conditions
    converted['Win conditions'] = ''
    converted.loc[recent_data['Extra Time'] == 1, 'Win conditions'] = 'extra time'
    converted.loc[recent_data['Penalty Shootout'] == 1, 'Win conditions'] = 'penalties'
    
    converted['Attendance'] = pd.NA
    converted['Half-time Home Goals'] = pd.NA
    converted['Half-time Away Goals'] = pd.NA
    converted['Referee'] = ''
    converted['Assistant 1'] = ''
    converted['Assistant 2'] = ''
    converted['RoundID'] = recent_data['Key Id'].values
    converted['MatchID'] = recent_data['Match Id'].str.replace('M-', '').str.replace('-', '').values
    converted['Home Team Initials'] = recent_data['Home Team Code']
    converted['Away Team Initials'] = recent_data['Away Team Code']
    
    # Read existing data
    existing_file = "WorldCupMatches.csv"
    if os.path.exists(existing_file):
        print(f"\nReading existing {existing_file}...")
        existing_df = pd.read_csv(existing_file)
        print(f"Existing data: {len(existing_df)} matches (up to {existing_df['Year'].max()})")
        
        # Remove any existing 2018/2022 data to avoid duplicates
        existing_df = existing_df[~existing_df['Year'].isin([2018, 2022])]
        print(f"After removing 2018/2022: {len(existing_df)} matches")
        
        # Combine
        combined_df = pd.concat([existing_df, converted], ignore_index=True)
    else:
        print(f"\n{existing_file} not found, creating new file...")
        combined_df = converted
    
    # Sort by year and save
    combined_df = combined_df.sort_values('Year').reset_index(drop=True)
    
    # Save backup of original
    if os.path.exists(existing_file):
        backup_file = "WorldCupMatches_backup.csv"
        print(f"\nCreating backup: {backup_file}")
        existing_df_original = pd.read_csv(existing_file)
        existing_df_original.to_csv(backup_file, index=False)
    
    # Save updated file
    output_file = "WorldCupMatches.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Successfully updated {output_file}")
    print(f"  Total matches: {len(combined_df)}")
    print(f"  Year range: {combined_df['Year'].min()} to {combined_df['Year'].max()}")
    print(f"  Tournaments: {sorted(combined_df['Year'].unique())}")
    
    print("\nSample of new data:")
    print(combined_df[combined_df['Year'].isin([2018, 2022])][['Year', 'Home Team Name', 'Home Team Goals', 'Away Team Goals', 'Away Team Name', 'Stage']].head(10))
    
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("FIFA World Cup Data Integration Tool")
    print("=" * 70)
    print("\nThis script will add 2018 and 2022 World Cup data to your dataset.")
    print("\nIMPORTANT: Make sure you have the file:")
    print("  'FIFA World Cup 1930-2022 All Match Dataset.csv'")
    print("in the current directory before running this script.")
    print("\n" + "=" * 70 + "\n")
    
    try:
        success = integrate_new_data()
        if success:
            print("\n" + "=" * 70)
            print("✓ Integration completed successfully!")
            print("=" * 70)
            print("\nNext steps:")
            print("1. Review the updated WorldCupMatches.csv file")
            print("2. Run your notebook to retrain the model with the new data")
            print("3. The backup file 'WorldCupMatches_backup.csv' contains your original data")
        else:
            print("\n✗ Integration failed. Please check the error messages above.")
    except Exception as e:
        print(f"\n✗ Error during integration: {e}")
        import traceback
        traceback.print_exc()
