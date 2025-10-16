import pandas as pd
from fuzzywuzzy import process

class FitnessRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.df["Difficulty Level_Cleaned"] = (
            self.df["Difficulty Level_Cleaned"].str.strip().str.capitalize()
        )
        self.df["Target Muscle Group_Cleaned"] = (
            self.df["Target Muscle Group_Cleaned"].fillna("").str.title().str.strip()
        )

    def fuzzy_match_muscle(self, target):
        all_muscles = self.df["Target Muscle Group_Cleaned"].dropna().unique()
        matches = process.extract(target, all_muscles, limit=3)
        return [m[0] for m in matches if m[1] > 70]

    def recommend(self, difficulty, muscle_group=None, n=5):
        df = self.df[self.df["Difficulty Level_Cleaned"] == difficulty]

        if muscle_group:
            similar = self.fuzzy_match_muscle(muscle_group)
            df = df[df["Target Muscle Group_Cleaned"].isin(similar)]

        if df.empty:
            df = self.df[self.df["Difficulty Level_Cleaned"] == difficulty]

        df = df.drop_duplicates(subset="Name of Exercise")
        q = df["Calories_Burned"].quantile([0.33, 0.66])

        light = df[df["Calories_Burned"] <= q[0.33]]
        medium = df[(df["Calories_Burned"] > q[0.33]) & (df["Calories_Burned"] <= q[0.66])]
        intense = df[df["Calories_Burned"] > q[0.66]]

        result = pd.concat([
            light.sample(n=min(1, len(light)), replace=False),
            medium.sample(n=min(2, len(medium)), replace=False),
            intense.sample(n=min(2, len(intense)), replace=False),
        ])

        return result.sample(frac=1).reset_index(drop=True)[[
            "Name of Exercise", "Target Muscle Group_Cleaned", "Difficulty Level_Cleaned", "Calories_Burned"
        ]]
