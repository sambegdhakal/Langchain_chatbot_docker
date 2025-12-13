import pandas as pd

df=pd.read_csv("datasets/commentary_data/commentary_2025_ENG.1.csv", nrows=5)
playerstats_df = pd.read_csv("datasets/playerStats_data/playerStats_2025_ENG.1.csv")
player_df = pd.read_csv("datasets/base_data/players.csv", low_memory=False)
team_df = pd.read_csv("datasets/base_data/teams.csv")
# Merge player stats with player info
merged_df = playerstats_df.merge(player_df, on="athleteId", how="left")
# Merge with team info
merged_df = merged_df.merge(team_df, on="teamId", how="left")

def test_commentary_file_load():
    assert not df.empty, "Empty file"
    
def test_field_in_commentary():
    assert "commentaryText" in df.columns, "Field not present in commentary file"

def test_player_stats_team_file_load():
    assert not playerstats_df.empty, "player stats file is empty"
    assert not player_df.empty, "player file is empty"
    assert not team_df.empty, "team file is empty"

def test_merge():
    assert not merged_df.empty, "merged_df is empty"
    required_cols = ["athleteId", "teamId"]
    for col in required_cols:
        assert col in merged_df.columns, f"{col} is missing in merged_df" 