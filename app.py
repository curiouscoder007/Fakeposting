from flask import Flask,render_template,request
import os
import numpy as np
from fakeposting import fakeposting
STATIC_DIR = os.path.abspath('../flaskproject64/static')
app = Flask(__name__)
app.static_folder = STATIC_DIR
print(STATIC_DIR)
print(app.root_path)
print(app.static_url_path)
@app.route("/", methods=['POST','GET'])
def intro():
  if request.method == 'POST':
    profile = request.form["profile"]
    commute = request.form["commute"]
    logo = request.form["logo"]
    questions = request.form["questions"]
    salary = request.form["salary"]
    education = request.form["education"]
    input_data = np.array([profile,commute, logo, questions, salary, education])
    model = fakeposting(input_data)
    output = model.predict()
    return render_template('intro.html',output=output)
  else:
    return render_template('intro.html',)

if __name__ == "__main__":
  app.run(debug=True)