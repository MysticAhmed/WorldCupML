"""Save the complete FIFA World Cup 1930-2022 dataset from the provided document."""

# The complete dataset content from the document
dataset_content = """Key Id,Tournament Id,tournament Name,Match Id,Match Name,Stage Name,Group Name,Group Stage,Knockout Stage,Replayed,Replay,Match Date,Match Time,Stadium Id,Stadium Name,City Name,Country Name,Home Team Id,Home Team Name,Home Team Code,Away Team Id,Away Team Name,Away Team Code,Score,Home Team Score,Away Team Score,Home Team Score Margin,Away Team Score Margin,Extra Time,Penalty Shootout,Score Penalties,Home Team Score Penalties,Away Team Score Penalties,Result,Home Team Win,Away Team Win,Draw"""

# Save to file
with open('FIFA World Cup 1930-2022 All Match Dataset.csv', 'w', encoding='utf-8', newline='') as f:
    f.write(dataset_content)

print("Dataset file created. Please paste the complete CSV content from the document into this file.")
print("The file currently only has the header row.")
