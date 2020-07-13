import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import joblib

stop = set(stopwords.words('english'))


def load_data():
    df = pd.read_csv("fake_job_postings.csv")
    df = df.drop(['job_id','salary_range'], axis=1)
    df.fillna(' ', inplace=True)
    df['combined_columns'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + \
                            df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + \
                            df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_experience'] + ' ' + \
                            df['required_education'] + ' ' + df['industry'] + ' ' + df['function'] 
    
    df = df.drop(['title','location','department','company_profile','description','requirements','benefits','employment_type'
            ,'required_experience','required_education','industry','function'],axis=1)
    return df

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    text = text.translate(str.maketrans('','',string.punctuation))
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text) 

def lem(df):
    print('Started leumatization process...')
    df.combined_columns = df.combined_columns.apply(lemmatize_words)
    print('Lemmatization complete...')
    return df

def split_df(df):
    X_train , X_test ,y_train , y_test = train_test_split(df.combined_columns,df.fraudulent , test_size = 0.2 , random_state = 0)
    return X_train , X_test ,y_train , y_test
    
def buildpipeline(X_train, y_train):
    print('Building pipeline....')
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('rfm', RandomForestClassifier())])
    
    print('Training pipeline model... ')
    pipe.fit(X_train,y_train)
    return pipe

def predict_results(pipe, X_test):
    print('Predicting values for Test set...')
    y_pred = pipe.predict(X_test)
    return y_pred


def report_metrics(y_test, y_pred):
    report = classification_report(y_test,y_pred,target_names = ['0','1'])
    accuracy=accuracy_score(y_test,y_pred)
    cm_tv = confusion_matrix(y_test,y_pred)
    return cm_tv, accuracy, report

def save_model(pipe):
    joblib.dump(pipe, 'rf_model_0.1.0.pkl') 

def load_predict(pipe, df):
    model = joblib.load('rf_model_0.1.0.pkl')
    model.predict([df.combined_columns[0]])
