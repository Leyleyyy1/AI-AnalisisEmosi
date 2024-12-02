import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import tkinter as tk
from nltk.corpus import stopwords

# Load and preprocess dataset
data = pd.read_csv("Twitter_Emotion_Dataset.csv")
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    return ' '.join(word for word in text.split() if word.lower() not in stop_words)

data['cleaned_text'] = data['tweet'].apply(clean_text)

# Train the model
X = data['cleaned_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Define the GUI
def analyze_emotion():
    input_text = text_entry.get("1.0", tk.END).strip()
    if input_text:
        cleaned_input = clean_text(input_text)
        probabilities = model.predict_proba([cleaned_input])[0]
        classes = model.classes_

        # Get the top 2 emotions with highest probabilities
        top_indices = probabilities.argsort()[-2:][::-1]  # Sort and take the top 2
        result_text = "Top Emotions:\n\n"
        for idx in top_indices:
            emotion = classes[idx].capitalize()
            prob = probabilities[idx] * 100
            result_text += f"{emotion}: {prob:.2f}%\n"
        
        result_label.config(text=result_text, fg="white")

# Set up the tkinter GUI with decorative elements
app = tk.Tk()
app.title("Emotion Analysis Tool")
app.geometry("400x400")
app.config(bg="#4B0082")

# Title Label
title_label = tk.Label(app, text="Emotion Analysis Tool", font=("Arial", 16, "bold"), bg="#4B0082", fg="white")
title_label.pack(pady=10)

# Text Entry with Label
input_label = tk.Label(app, text="Enter text for emotion analysis:", font=("Arial", 12), bg="#4B0082", fg="white")
input_label.pack(pady=5)
text_entry = tk.Text(app, height=5, width=40, font=("Arial", 12))
text_entry.pack(pady=10)

# Analyze Button with styling
analyze_button = tk.Button(app, text="Analyze Emotion", font=("Arial", 12, "bold"), command=analyze_emotion, bg="#8A2BE2", fg="white", relief="groove")
analyze_button.pack(pady=10)

# Result Label for displaying emotion probabilities
result_label = tk.Label(app, text="", font=("Arial", 12), bg="#4B0082", justify="left")
result_label.pack(pady=10)

# Run the application
app.mainloop()
