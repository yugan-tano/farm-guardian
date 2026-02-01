# app.py - 复制这个完整代码
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
import random

app = Flask(__name__)

# 配置
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 病害数据库
DISEASE_DB = {
    'healthy': {'name': '健康', 'risk': '低', 'confidence': '92.5%'},
    'ulcer': {'name': '猕猴桃溃疡病', 'risk': '高', 'confidence': '87.3%'},
    'brown_spot': {'name': '猕猴桃褐斑病', 'risk': '中', 'confidence': '78.9%'},
    'flower_rot': {'name': '猕猴桃花腐病', 'risk': '高', 'confidence': '85.6%'}
}

# 处理建议
ADVICE_DB = {
    'healthy': '作物健康状况良好，继续保持当前管理措施。',
    'ulcer': '1.立即隔离病株\n2.喷洒50%多菌灵800倍液，连续3天\n3.清除病残体，减少病原',
    'brown_spot': '1.剪除病叶集中销毁\n2.喷洒70%代森锰锌600倍液\n3.改善通风透光条件',
    'flower_rot': '1.摘除病花\n2.喷洒50%速克灵1500倍液\n3.增施磷钾肥，提高抗病力'
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传图片API"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有选择文件'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '文件格式不支持'})

    # 保存文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 模拟AI识别（为了演示，固定返回溃疡病）
    # 你可以看到效果，稍后替换为真实AI
    disease_key = 'ulcer'  # 固定为溃疡病用于演示

    disease_info = DISEASE_DB[disease_key]

    # 生成报告
    report = {
        'success': True,
        'filename': filename,
        'disease': disease_info['name'],
        'confidence': disease_info['confidence'],
        'risk_level': disease_info['risk'],
        'advice': ADVICE_DB[disease_key],
        'image_url': f'/static/uploads/{filename}',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'report_id': f"RPT{timestamp}"
    }

    return jsonify(report)


@app.route('/api/demo')
def demo_data():
    """演示数据API"""
    diseases = list(DISEASE_DB.keys())

    # 生成5条演示记录
    records = []
    for i in range(5):
        disease_key = random.choice(diseases)
        disease_info = DISEASE_DB[disease_key]

        # 生成一个过去的时间
        time_delta = random.randint(1, 24)
        past_time = datetime.now().replace(hour=10, minute=30)

        records.append({
            'id': i + 1,
            'disease': disease_info['name'],
            'confidence': disease_info['confidence'],
            'risk_level': disease_info['risk'],
            'timestamp': past_time.strftime('%Y-%m-%d %H:%M'),
            'advice': ADVICE_DB[disease_key][:50] + '...'
        })

    return jsonify({'records': records})


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """访问上传的图片"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': '智农虫盾',
        'version': '1.0.0',
        'time': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🚜 智农虫盾 - 病虫害监测系统")
    print("=" * 50)
    print("🌐 访问地址: http://localhost:5000")
    print("🔧 健康检查: http://localhost:5000/health")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)