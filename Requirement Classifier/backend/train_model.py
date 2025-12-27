import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
df = pd.read_excel("FR_NFR_Dataset.xlsx")
df.dropna(inplace=True)

X = df["Requirement Text"]
y = df["Type"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model training complete and saved as model.pkl")
