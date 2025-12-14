import pandas as pd

# Load CSVs
df_playerstats = pd.read_csv("datasets/playerStats_data/playerStats_2025_ENG.1.csv")
df_players = pd.read_csv("datasets/base_data/players.csv", low_memory=False)
df_teams = pd.read_csv("datasets/base_data/teams.csv")
df_team_standings = pd.read_csv("datasets/base_data/standings.csv")
df_venue = pd.read_csv("datasets/base_data/venues.csv")
df_leagues= pd.read_csv("datasets/base_data/leagues.csv")



# Merge player stats with player info
merged_df = df_playerstats.merge(df_players, on="athleteId", how="left")
merged_df = merged_df.merge(df_teams, on="teamId", how="left")

# Sort team standings by teamId and timeStamp descending
df_sorted = df_team_standings.sort_values(["teamId", "timeStamp"], ascending=[True, False])
df_standings_selected = df_sorted.drop_duplicates(subset=["teamId"], keep="first")

# Select only needed columns
df_standings_selected = df_standings_selected[["form", "next_opponent", "teamId", "next_homeAway", "next_matchDateTime", "timeStamp","seasonType"]]

df_teams_selected = df_teams[["name", "abbreviation", "teamId", "venueId"]].rename(
    columns={"name": "team_name", "abbreviation": "team_abbreviation"}
)

df_venue_selected = df_venue[["fullName", "venueId"]].rename(columns={"fullName": "stadium_name"})

df_leagues_selected= df_leagues[["seasonType", "seasonName"]]

# Merge team standings with team and venue info
merged_df_standings = (
    df_teams_selected
    .merge(df_standings_selected, on="teamId", how="left")
    .merge(df_venue_selected, on="venueId", how="left")
)

# Add next opponent name
team_id_to_name = df_teams_selected.set_index("teamId")["team_name"].to_dict()
merged_df_standings["next_opponent_name"] = merged_df_standings["next_opponent"].map(team_id_to_name)

# Drop unnecessary columns
merged_df_standings = merged_df_standings.drop(columns=["venueId", "teamId", "next_opponent", "timeStamp"])

merged_df_standings_final= merged_df_standings.merge(df_leagues_selected, on="seasonType", how="left")

# Drop the seasonType column
merged_df_standings_final = merged_df_standings_final.drop(columns=["seasonType"])

# Filter rows
merged_df_standings_final = merged_df_standings_final[
    merged_df_standings_final["seasonName"].str.contains("English Premier League", case=False) &
    merged_df_standings_final["seasonName"].str.contains("25-26", case=False)
]

# Display final DataFrame
print("===== Merged Team Standings =====")
print(merged_df_standings_final)

# Optionally, save to Excel
merged_df_standings_final.to_excel("merged_df_standings.xlsx", index=False)
print("\nSaved merged_df_standings.xlsx")
