"""
Save the new FIFA World Cup dataset from the provided data.
This script creates the CSV file from the document content.
"""

# The dataset content - first few rows as header
dataset_content = """Key Id,Tournament Id,tournament Name,Match Id,Match Name,Stage Name,Group Name,Group Stage,Knockout Stage,Replayed,Replay,Match Date,Match Time,Stadium Id,Stadium Name,City Name,Country Name,Home Team Id,Home Team Name,Home Team Code,Away Team Id,Away Team Name,Away Team Code,Score,Home Team Score,Away Team Score,Home Team Score Margin,Away Team Score Margin,Extra Time,Penalty Shootout,Score Penalties,Home Team Score Penalties,Away Team Score Penalties,Result,Home Team Win,Away Team Win,Draw
1,WC-1930,1930 FIFA World Cup,M-1930-01,France v Mexico,group stage,Group 1,1,0,0,0,7/13/1930,15:00,S-193,Estadio Pocitos,Montevideo,Uruguay,T-28,France,FRA,T-44,Mexico,MEX,4�1,4,1,3,-3,0,0,0-0,0,0,home team win,1,0,0"""

print("Please manually copy the dataset from the document into a file named:")
print("'FIFA World Cup 1930-2022 All Match Dataset.csv'")
print("\nThe file should be placed in the WorldCupML directory.")
print("\nOnce you've done that, run:")
print("  python convert_new_dataset.py")
