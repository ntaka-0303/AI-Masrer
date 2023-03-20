# import
import os, shutil
from flask import Flask, render_template, request, redirect, url_for, redirect, Markup
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

# define the static variables
UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# difine the label and hyperparameters
labels = ["Tシャツ・トップス", "パンツ", "プルオーバー", "ワンピース", "コート", "サンダル", "シャツ", "スニーカー", "バッグ", "アンクルブーツ"]
n_class = len(labels)
img_size = 28
n_result = 3 # the number of results to be displayed

# app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# define the function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# display the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

# display the result page
@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        # check file is exist and allowed
        if 'file' not in request.files:
            print('file not uploaded')
            return redirect(url_for('index'))
        file = request.files['file']
        if not allowed_file(file.filename):
            print('file not allowed')
            return redirect(url_for('index'))
        
        # save the file
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

        # read the image
        image = Image.open(os.path.join(UPLOAD_FOLDER, filename))
        image = image.convert('L')
        image = image.resize((img_size, img_size))
        data = np.asarray(image, dtype=np.float32)
        data = data.reshape(1, img_size, img_size, 1) / 255.0

        # predict
        model = tf.keras.models.load_model('./model/fashion_classifier.h5')
        predicted = model.predict(data)[0]
        sorted_idx = np.argsort(predicted)[::-1] # sort the result in descending order
        result = ""
        for i in range(n_result):
            idx = sorted_idx[i]
            raito = predicted[idx]
            label = labels[idx]
            result += "<p>" + str(round(raito * 100, 2)) + "%の確率で " + label + "です。 </p>"
        return render_template('result.html', result=Markup(result), filename=filename)
    else:
        return redirect(url_for('index'))
    
# run the app
if __name__ == '__main__':
    app.run(debug=True)