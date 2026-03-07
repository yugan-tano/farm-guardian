# app.py (最终修复版)
import os
import uuid
import datetime
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
# 引入我们的核心 AI 模块
from core.predictor import Predictor

# --- 初始化 Flask App ---
app = Flask(__name__)

# --- 配置 ---
# 上传文件夹
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- AI 引擎初始化 ---
print("正在初始化 AI 核心...")
ai_engine = Predictor()
print("✅ AI 核心准备就绪。")


def allowed_file(filename):
    """检查文件后缀名是否合法"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- API 路由定义 ---

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_and_predict():
    """处理文件上传和AI预测"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有上传文件'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'})

    if file and allowed_file(file.filename):
        try:
            # 1. 保存文件，使用 uuid 防止重名
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # 2. 调用 AI 进行推理
            result = ai_engine.predict(filepath, crop_type='kiwi')

            # 3. 构造返回给前端的数据
            confidence_value = result['confidence']  # 这是个 float, e.g. 92.1

            # 【已修复】直接用 float 进行数值比较
            if confidence_value > 80:
                risk_level = '高'
            elif confidence_value > 50:
                risk_level = '中'
            else:
                risk_level = '低'

            # 准备要返回给前端的 response
            response_data = {
                'success': True,
                'disease': result['class_name'],
                'confidence': f"{confidence_value:.1f}%",  # 格式化成带 '%' 的字符串给前端显示
                'risk_level': risk_level,
                'advice': f"初步诊断为 {result['class_name']}。建议采取农业防治措施，如清除病残体；并可使用对应的化学药剂进行防治，具体请咨询农技专家。",
                'image_url': url_for('static', filename=f'uploads/{unique_filename}'),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'report_id': str(uuid.uuid4())[:8].upper()
            }
            return jsonify(response_data)

        except Exception as e:
            # 打印详细错误到后台，方便调试
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'服务器内部错误: {e}'})

    return jsonify({'success': False, 'error': '文件类型不支持'})


# 【新增】为前端提供一个假的“历史记录”API，防止报错
@app.route('/api/demo')
def demo_history():
    """提供演示用的历史数据"""
    demo_records = {
        "records": [
            {
                "disease": "猕猴桃溃疡病",
                "risk_level": "高",
                "confidence": "92.1%",
                "timestamp": "2023-11-20 09:15"
            },
            {
                "disease": "健康叶片",
                "risk_level": "低",
                "confidence": "98.8%",
                "timestamp": "2023-11-19 14:30"
            }
        ]
    }
    return jsonify(demo_records)


# 【新增】为前端提供一个“健康检查”API，防止报错
@app.route('/health')
def health_check():
    """系统健康检查接口"""
    return jsonify({'status': 'ok'})


# --- 启动器 ---
if __name__ == '__main__':
    # 启动应用，host='0.0.0.0' 允许局域网其他设备访问
    app.run(host='0.0.0.0', port=5000, debug=True)