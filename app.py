from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

app = Flask(__name__)

def train_chatbot(dataset_path):
    df = pd.read_csv(dataset_path)

    # Applying oversampling to handle class imbalance
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(df['questions'].values.reshape(-1, 1), df['category'])

    # Menggunakan TfidfVectorizer untuk membuat matriks TF-IDF dari dataset oversampled
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(X_resampled.flatten())

    # Menyiapkan model klasifikasi Naive Bayes
    classifier = MultinomialNB()
    classifier.fit(tfidf_matrix, y_resampled)

    return df, vectorizer, classifier

def get_response(user_input, df, vectorizer, classifier):
    # Menghitung cosine similarity antara input pengguna dan setiap pertanyaan dalam dataset
    input_vector = vectorizer.transform([user_input])

    # Memprediksi kategori jawaban menggunakan model Naive Bayes
    predicted_category = classifier.predict(input_vector)

    # Memfilter dataset berdasarkan kategori yang diprediksi
    filtered_df = df[df['category'] == predicted_category[0]]

    # Menghitung cosine similarity antara input dan pertanyaan di kategori yang diprediksi
    tfidf_matrix_filtered = vectorizer.transform(filtered_df['questions'].values.astype('U'))
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix_filtered).flatten()

    # Mendapatkan indeks pertanyaan dengan similarity tertinggi
    max_similarity_index = cosine_similarities.argmax()

    # Mengembalikan jawaban yang sesuai
    response = filtered_df['answer'].iloc[max_similarity_index]
    return response

# Gantilah 'knowledgebase.csv' dengan nama file CSV dataset Anda
dataset_path = 'knowledgebase.csv'
df, vectorizer, classifier = train_chatbot(dataset_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    response = get_response(user_input, df, vectorizer, classifier)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
