import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from config import Config

# 初始化 Flask 应用
app = Flask(__name__)
app.config.from_object(Config)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """检查文件后缀是否合法"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET'])
def index():
    """渲染主页"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API接口：接收图片和作物类型，返回预测结果
    """
    # 1. 校验请求中是否有文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    crop_type = request.form.get('crop_type', 'kiwi')  # 默认为猕猴桃

    # 2. 校验文件内容
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    # 3. 保存文件 (为了安全，使用 secure_filename)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # TODO: 这里即将调用 core/predictor.py 进行真实预测
        # 目前先返回模拟数据，确保前后端跑通
        print(f"[System] Processing {crop_type} image: {filepath}")

        # 模拟结果
        result = {
            'class_name': 'ulcer',  # 溃疡病
            'confidence': 0.985,  # 置信度
            'crop': crop_type
        }
        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 启动服务，开启 Debug 模式方便调试
    app.run(host='0.0.0.0', port=5000, debug=True)