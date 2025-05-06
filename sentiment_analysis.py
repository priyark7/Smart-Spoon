
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample feedback and labels (1 = positive, 0 = negative)
feedback = [
    "I love the spoon! Tastes feel enhanced",
    "Didn't feel much difference",
    "Amazing product for low-sodium diets",
    "Disappointed with the effect"
]
labels = [1, 0, 1, 0]

# Vectorize and train
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(feedback)

clf = MultinomialNB()
clf.fit(X, labels)

# Predict sentiment for new feedback
new_feedback = ["It really improved my meals"]
X_new = vectorizer.transform(new_feedback)
predicted_sentiment = clf.predict(X_new)

print("Positive" if predicted_sentiment[0] == 1 else "Negative")
