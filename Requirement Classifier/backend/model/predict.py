from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Dummy training data
X_train = [
    "System shall allow user to login",
    "System should be available 99% of the time",
    "User should be able to reset password",
    "Response time should be less than 1 second"
]
y_train = ["functional", "non_functional", "functional", "non_functional"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_vec, y_train)

def classify_requirement(text):
    X_input = vectorizer.transform([text])
    return model.predict(X_input)[0]
