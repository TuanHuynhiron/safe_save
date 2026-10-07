import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Cấu hình danh mục cây trồng hỗ trợ
CROPS_INFO = {
    'ca-chua': {'name': 'Cà Chua', 'icon': '🍅', 'badge': 'bg-danger'},
    'ca-rot': {'name': 'Cà Rốt', 'icon': '🥕', 'badge': 'bg-warning text-dark'}
}

def get_crop_paths(crop_type):
    """Tạo & trả về đường dẫn dataset, model và classes riêng cho từng loại cây"""
    dataset_dir = os.path.join("dataset", crop_type)
    models_dir = "models"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{crop_type}_model.pth")
    classes_path = os.path.join(models_dir, f"{crop_type}_classes.json")
    return dataset_dir, model_path, classes_path

# Biến đổi ảnh chuẩn cho PyTorch
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_model(num_classes):
    """Khởi tạo MobileNetV2"""
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# --- ROUTES GIAO DIỆN ---

@app.route('/')
def index():
    """Trang chủ: Chọn khu vườn"""
    return render_template('index.html', crops=CROPS_INFO)

@app.route('/plant/<crop_type>')
def plant_page(crop_type):
    """Trang chẩn đoán & train chuyên biệt cho từng loại cây"""
    if crop_type not in CROPS_INFO:
        return "Loại cây không hợp lệ!", 404
    return render_template('plant.html', crop_type=crop_type, crop=CROPS_INFO[crop_type])

# --- API ENDPOINTS (Phân tách theo crop_type) ---

@app.route('/api/<crop_type>/dataset-info', methods=['GET'])
def dataset_info(crop_type):
    dataset_dir, _, _ = get_crop_paths(crop_type)
    classes_info = {}
    if os.path.exists(dataset_dir):
        for sub in os.listdir(dataset_dir):
            sub_path = os.path.join(dataset_dir, sub)
            if os.path.isdir(sub_path):
                imgs = glob.glob(os.path.join(sub_path, "*.*"))
                classes_info[sub] = len(imgs)
    return jsonify({"classes": classes_info})

@app.route('/api/<crop_type>/add-sample', methods=['POST'])
def add_sample(crop_type):
    dataset_dir, _, _ = get_crop_paths(crop_type)
    disease_name = request.form.get('disease_name', '').strip()
    files = request.files.getlist('files')

    if not disease_name or not files or len(files) == 0:
        return jsonify({'error': 'Vui lòng nhập tên bệnh và chọn ít nhất 1 hình ảnh!'}), 400

    safe_name = disease_name.replace("/", "_").replace("\\", "_")
    class_dir = os.path.join(dataset_dir, safe_name)
    os.makedirs(class_dir, exist_ok=True)

    saved_count = 0
    existing_count = len(os.listdir(class_dir))

    for idx, file in enumerate(files):
        if file and file.filename != '':
            filename = f"{existing_count + idx + 1}.jpg"
            filepath = os.path.join(class_dir, filename)
            try:
                image = Image.open(file.stream).convert('RGB')
                image.save(filepath)
                saved_count += 1
            except Exception:
                continue

    return jsonify({'message': f'Đã tải lên thành công {saved_count} ảnh vào danh mục "{safe_name}"!'})

@app.route('/api/<crop_type>/train', methods=['POST'])
def train_model(crop_type):
    dataset_dir, model_path, classes_path = get_crop_paths(crop_type)
    epochs = int(request.form.get('epochs', 5))
    
    classes = [d for d in os.listdir(dataset_dir) 
               if os.path.isdir(os.path.join(dataset_dir, d)) 
               and len(os.listdir(os.path.join(dataset_dir, d))) > 0]
    
    if len(classes) < 2:
        return jsonify({
            'error': f'Cần ít nhất 2 nhãn bệnh khác nhau để AI huấn luyện phân biệt cho cây {CROPS_INFO[crop_type]["name"]}!'
        }), 400

    try:
        dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transforms)
        dataloader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

        class_names = dataset.classes
        with open(classes_path, 'w', encoding='utf-8') as f:
            json.dump(class_names, f, ensure_ascii=False)

        model = build_model(len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        model.train()
        logs = []

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = (correct / total) * 100
            log_str = f"Epoch {epoch+1}/{epochs} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.1f}%"
            logs.append(log_str)

        torch.save(model.state_dict(), model_path)

        return jsonify({
            'message': f'Huấn luyện AI cho {CROPS_INFO[crop_type]["name"]} thành công!',
            'logs': logs
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi huấn luyện: {str(e)}'}), 500

@app.route('/api/<crop_type>/predict', methods=['POST'])
def predict(crop_type):
    dataset_dir, model_path, classes_path = get_crop_paths(crop_type)
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Vui lòng chọn ảnh để chẩn đoán!'}), 400

    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return jsonify({'error': f'Chưa có mô hình AI cho {CROPS_INFO[crop_type]["name"]}! Vui lòng thêm dữ liệu và huấn luyện trước.'}), 400

    with open(classes_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    model = build_model(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    image = Image.open(file.stream).convert('RGB')
    tensor = eval_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)

    return jsonify({
        'disease': predicted_class,
        'confidence': confidence_pct
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)