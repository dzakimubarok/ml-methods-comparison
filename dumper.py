import dataset
from datafreeze import freeze

db = dataset.connect("sqlite:///tweets.db")
result = db['myTable'].all()
freeze(result, format='csv', filename='tweets.csv')