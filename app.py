from flask import Flask, request, jsonify, render_template
import pandas as pd
from recommendation_system import get_recommendations

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.form
    movie_title = data['title']
    recommendations = get_recommendations(movie_title)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
