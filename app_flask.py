from flask import Flask, request, render_template, send_file
from flask_cors import CORS
import qrcode
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/get_qrcode',methods=['GET', 'POST'])
def home():
    """生成二维码，指向上传页面"""
    print('testset')
    upload_url = "http://localhost:5000/upload"  # 服务器部署后需替换为公网 URL
    qr = qrcode.make(upload_url)
    qr_path = os.path.join(UPLOAD_FOLDER, 'upload_qr.png')
    qr.save(qr_path)
    return send_file(qr_path, mimetype='image/png')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """文件上传页面"""
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            return "文件上传成功！"
    
    return '''
    <!doctype html>
    <html>
    <body>
        <h2>上传图片</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="上传">
        </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
