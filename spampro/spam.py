

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
import re
from nltk.corpus import stopwords

# 1. Load the dataset (replace 'spam.csv' with your file path)
# The dataset is often encoded with 'latin-1' and has no header, separated by tabs.
data = pd.read_csv(
    r'C:\Users\surthika\Downloads\spam.csv', 
    encoding='latin-1', 
    sep='\t',
    header=None # <-- ADD THIS LINE
)


# 2. Drop irrelevant columns (usually the last three)
#data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

# 3. Rename columns for simplicity
data = data.rename(columns={0: 'Label', 1: 'Message'})

# 4. Remove duplicate messages
data = data.drop_duplicates(keep='first')

# 5. Encode the target variable ('ham' and 'spam') to numerical (0 and 1)
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

print("First 5 rows of the cleaned data:")
print(data.head())
print(f"Total entries: {len(data)}")

# Download stopwords list if not present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove all non-word characters (punctuation, numbers)
    text = re.sub(r'\W', ' ', text)
    
    # 3. Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    
    # 4. Remove multiple spaces
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    # 5. Remove stop words and join the words back
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply the cleaning function to the 'Message' column
data['Cleaned_Message'] = data['Message'].apply(clean_text)

print("\nExample of cleaned message:")
print(data['Cleaned_Message'].iloc[2])

X = data['Cleaned_Message']  # Features
y = data['Label']            # Target

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# 1. Fit the vectorizer on the TRAINING data (learn the vocabulary)
X_train_vec = vectorizer.fit_transform(X_train)

# 2. Transform the TEST data using the vocabulary learned from training
X_test_vec = vectorizer.transform(X_test)

print(f"Shape of training features (messages x unique words): {X_train_vec.shape}")

# Initialize the Multinomial Naive Bayes model
mnb = MultinomialNB()

# Train the model
mnb.fit(X_train_vec, y_train)

print("\nMultinomial Naive Bayes model trained successfully!")

# Make predictions on the test set
y_pred = mnb.predict(X_test_vec)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#  to test a sample message
sample_msg = "Your credit card account has been SUSPENDED due to unusual activity. Reply YES to this message or call 0800123456 immediately to re-activate."


cleanedsample = clean_text(sample_msg)

samplevec = vectorizer.transform([cleanedsample])

prediction = mnb.predict(samplevec)[0]

result = "SPAM" if prediction == 1 else "HAM"

print(f"\nSample Message: '{sample_msg}'")
print(f"Prediction: This message is {result}")