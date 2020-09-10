import nltk
import pandas as pd
import matplotlib as plt
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))
ps = PorterStemmer()
lmtzr = WordNetLemmatizer()
print(stop)

df = pd.read_csv('testdata/testdatatest.csv')
sent = (df['text'])
# print(sent)

# tokenize l                      sent.apply(lambda x: nltk.word_tokenize(x))
# less than 2 k                   sent.str.findall('\w{3,}').str.join(' ')
# stopwords removing j            sent.apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
# lemmatizing i                   sent.apply(lambda x:' '.join([lmtzr.lemmatize(word,'v') for word in x.split() ]))
# cleaning--------------------------------------------------------------------------
# punct   h                       sent.str.replace('[^\w]', ' ', case=False)
# non ascii shit   g              sent.str.replace('[^\x00-\x7F]+', '', case=False)
# @user f                         sent.str.replace('@[\w_]+', '', case=False)
# rt @user e                      sent.str.replace('rt @[\w_]+', '', case=False)
# hashtag   d                     sent.str.replace('#[\w_]+', '', case=False)
# numbers  c                      sent.str.replace('\d+', '', case=False)
# url  b                          sent.str.replace('http\S+|www.\S+', '', case=False)
# cleaning--------------------------------------------------------------------------
# casefolding a                   sent.apply(lambda x: ' '.join([word.lower() for word in x.split()]))


# new_sent = sent.apply(lambda x: nltk.word_tokenize(x))
# print(new_sent)
#
# new_sent.to_csv('testdata/Step L.csv', index=True, header='text')


