"""
Convert the FIFA World Cup 1930-2022 All Match Dataset to the format expected by the predictor.
This script reads the new comprehensive dataset and converts it to match the existing data structure.
"""

import pandas as pd
import sys

def convert_dataset(input_file: str, output_file: str):
    """
    Convert the new dataset format to the existing WorldCupMatches.csv format.
    
    Args:
        input_file: Path to the new CSV file
        output_file: Path to save the converted CSV
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df['Match Date'].min()} to {df['Match Date'].max()}")
    
    # Create the converted dataframe with mapped columns
    converted = pd.DataFrame()
    
    # Parse the date to extract year and create datetime
    df['Match Date'] = pd.to_datetime(df['Match Date'], format='%m/%d/%Y', errors='coerce')
    
    # Map columns from new format to old format
    converted['Year'] = df['Match Date'].dt.year
    
    # Combine date and time for Datetime column
    converted['Datetime'] = df['Match Date'].dt.strftime('%d %b %Y') + ' - ' + df['Match Time'].fillna('00:00')
    
    # Map stage names
    stage_mapping = {
        'group stage': 'Group Stage',
        'round of 16': 'Round of 16',
        'quarter-finals': 'Quarter-finals',
        'semi-finals': 'Semi-finals',
        'final': 'Final',
        'third-place match': 'Third Place',
    }
    converted['Stage'] = df['Stage Name'].str.lower().map(stage_mapping).fillna(df['Stage Name'])
    
    # Stadium and location
    converted['Stadium'] = df['Stadium Name']
    converted['City'] = df['City Name']
    
    # Team names and scores
    converted['Home Team Name'] = df['Home Team Name']
    converted['Home Team Goals'] = df['Home Team Score'].astype(float)
    converted['Away Team Goals'] = df['Away Team Score'].astype(float)
    converted['Away Team Name'] = df['Away Team Name']
    
    # Win conditions (extra time or penalties)
    converted['Win conditions'] = ''
    converted.loc[df['Extra Time'] == 1, 'Win conditions'] = 'extra time'
    converted.loc[df['Penalty Shootout'] == 1, 'Win conditions'] = 'penalties'
    
    # Attendance - not in new dataset, set to NaN
    converted['Attendance'] = pd.NA
    
    # Half-time scores - not in new dataset, set to NaN
    converted['Half-time Home Goals'] = pd.NA
    converted['Half-time Away Goals'] = pd.NA
    
    # Officials - not in new dataset, set to empty
    converted['Referee'] = ''
    converted['Assistant 1'] = ''
    converted['Assistant 2'] = ''
    
    # IDs
    converted['RoundID'] = df['Key Id']
    converted['MatchID'] = df['Match Id'].str.replace('M-', '').str.replace('-', '')
    
    # Team initials
    converted['Home Team Initials'] = df['Home Team Code']
    converted['Away Team Initials'] = df['Away Team Code']
    
    # Remove any rows with missing critical data
    critical_cols = ['Year', 'Home Team Name', 'Away Team Name', 'Home Team Goals', 'Away Team Goals']
    before_drop = len(converted)
    converted = converted.dropna(subset=critical_cols)
    after_drop = len(converted)
    
    if before_drop > after_drop:
        print(f"Dropped {before_drop - after_drop} rows with missing critical data")
    
    # Sort by date
    converted = converted.sort_values('Year').reset_index(drop=True)
    
    print(f"\nConverted dataset: {len(converted)} rows")
    print(f"Year range: {converted['Year'].min()} to {converted['Year'].max()}")
    print(f"Tournaments covered: {sorted(converted['Year'].unique())}")
    
    # Save to CSV
    converted.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    # Show sample
    print("\nSample of converted data:")
    print(converted[['Year', 'Home Team Name', 'Home Team Goals', 'Away Team Goals', 'Away Team Name', 'Stage']].head(10))
    
    return converted

if __name__ == "__main__":
    input_file = "FIFA World Cup 1930-2022 All Match Dataset.csv"
    output_file = "WorldCupMatches_Updated.csv"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        convert_dataset(input_file, output_file)
        print("\n✓ Conversion completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
