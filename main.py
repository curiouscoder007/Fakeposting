from flask import Flask  
from fake_job_postings import *
import joblib
import pandas as pd

app = Flask(__name__)


@app.route("/")             
def index():
    df = load_data()
    if Training:
        df = lem(df)
        X_train , X_test ,y_train , y_test = split_df(df)
        pipe = buildpipeline(X_train, y_train)
        y_pred = predict_results(pipe, X_test)
        save_model(pipe)
        cm_tv, accuracy, report = report_metrics(y_test, y_pred)
        return 'classification report : ', report
    else:
        for i in df.combined_columns:
            if len(str(i)) > 3:
                posting = lemmatize_words(i)
                break
        model = joblib.load('rf_model_0.1.0.pkl')
        result = model.predict([posting])
        if result == 0:
            return 'Fake job posting'
        else:
            return 'Genuine job posting'

@app.route("/healthcheck")             
def healthcheck():
    return {'Health of API':'OK'}
    

if __name__ == "__main__":
    Training = True
    app.run()