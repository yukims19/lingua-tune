from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


""" Define Model """
data = pd.read_csv(
    "https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
x = np.array(data['Text'])
y = np.array(data['language'])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)


app = Flask(__name__)


@app.route('/', methods=('GET', 'POST'))
def index():
    result = None

    if request.method == 'POST':
        name_val = request.form['content']
        data = cv.transform([name_val]).toarray()
        result = model.predict(data)

    return render_template('index.html', languages=",".join(result))
