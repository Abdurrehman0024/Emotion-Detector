import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data (semicolon-separated)
# train_df = pd.read_csv('data/train.txt', sep=';', header=None, names=['text', 'label'])
df1 = pd.read_csv('data/train.txt', sep=';', header=None, names=['text', 'label'])
df2 = pd.read_csv('data/genz_slangs.txt', sep=';', header=None, names=['text', 'label'])
train_df = pd.concat([df1, df2], ignore_index=True)

# Preview data (optional)
print(train_df.head())

# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(train_df['text'], train_df['label'])

# Save the model
joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as model.pkl")
