import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def import_data(path):
    try:
        with open(path) as file:
            recipes = json.load(file)
    except:
        print("File not found")
        exit(1)
    recipe_df = pd.DataFrame.from_dict(recipes)
    # convert list to string
    recipe_df["ingredients"] = recipe_df["ingredients"].apply(lambda x: " ".join(x))
    return recipe_df

def tfidf(train_df, test_df):
    vectorizer = TfidfVectorizer()
    recipe_tfidf = vectorizer.fit_transform(train_df)
    test_tfidf = vectorizer.transform(test_df)
    return recipe_tfidf, test_tfidf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: recipes.py [path]")
        exit(1)
    recipes = import_data(sys.argv[1])

    # split the dataset for cross validation later
    train_ingredients, test_ingredients, train_cuisine, test_cuisine = train_test_split(recipes["ingredients"], recipes["cuisine"],
                                                                                        test_size=.25, random_state=0)

    recipe_tfidf, test_tfidf = tfidf(train_ingredients, test_ingredients)
    # logistic regression
    logistic_clf = LogisticRegression()
    logistic_clf.fit(recipe_tfidf, train_cuisine)
    predicted_cuisine = logistic_clf.predict(test_tfidf)
    print("Logistic Regression Accuracy: " + str(accuracy_score(test_cuisine, predicted_cuisine)))
    # multinomial NB
    nb_clf = MultinomialNB().fit(recipe_tfidf, train_cuisine)
    predicted_cuisine = nb_clf.predict(test_tfidf)
    print("MultinomialNB Accuracy: " + str(accuracy_score(test_cuisine, predicted_cuisine)))
    # kNN

    # k means

    # svm
    svm_clf = LinearSVC()
    svm_clf.fit(recipe_tfidf, train_cuisine)
    predicted_cuisine = svm_clf.predict(test_tfidf)
    print("SVM Accuracy: " + str(accuracy_score(test_cuisine, predicted_cuisine)))
    # random forest

    # nueral network
