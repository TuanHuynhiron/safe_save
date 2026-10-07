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

# Cấu hình các đường dẫn lưu trữ
DATASET_DIR = "dataset"
MODEL_PATH = "leaf_model.pth"
CLASSES_PATH = "classes.json"
os.makedirs(DATASET_DIR, exist_ok=True)

# Biến đổi ảnh để đưa vào AI (Augmentation & Normalization)
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
    """Khởi tạo mô hình MobileNetV2 và thay lớp classifier cuối theo số lớp bệnh"""
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset-info', methods=['GET'])
def dataset_info():
    """Lấy danh sách các lớp bệnh và số lượng ảnh trong tập dữ liệu"""
    classes_info = {}
    if os.path.exists(DATASET_DIR):
        for sub in os.listdir(DATASET_DIR):
            sub_path = os.path.join(DATASET_DIR, sub)
            if os.path.isdir(sub_path):
                imgs = glob.glob(os.path.join(sub_path, "*.*"))
                classes_info[sub] = len(imgs)
    return jsonify({"classes": classes_info})

@app.route('/api/add-sample', methods=['POST'])
def add_sample():
    """Tải NGUYÊN BỘ NHIỀU ẢNH MẪU cùng lúc vào thư mục nhãn bệnh tương ứng"""
    disease_name = request.form.get('disease_name', '').strip()
    files = request.files.getlist('files') # Lấy danh sách nhiều file gửi lên

    if not disease_name or not files or len(files) == 0:
        return jsonify({'error': 'Vui lòng nhập tên bệnh và chọn ít nhất 1 hình ảnh!'}), 400

    # Chuẩn hóa tên thư mục nhãn
    safe_name = disease_name.replace("/", "_").replace("\\", "_")
    class_dir = os.path.join(DATASET_DIR, safe_name)
    os.makedirs(class_dir, exist_ok=True)

    saved_count = 0
    existing_count = len(os.listdir(class_dir))

    # Lặp qua từng file và lưu vào đĩa
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

@app.route('/api/train', methods=['POST'])
def train_model():
    """Huấn luyện lại mô hình AI bằng PyTorch (Fine-Tuning qua từng Epoch)"""
    epochs = int(request.form.get('epochs', 5))
    
    classes = [d for d in os.listdir(DATASET_DIR) 
               if os.path.isdir(os.path.join(DATASET_DIR, d)) 
               and len(os.listdir(os.path.join(DATASET_DIR, d))) > 0]
    
    if len(classes) < 2:
        return jsonify({
            'error': 'Cần ít nhất 2 nhãn bệnh khác nhau (ví dụ: "Đốm lá đậu phộng" và "Lá khỏe mạnh") để AI huấn luyện phân biệt!'
        }), 400

    try:
        dataset = datasets.ImageFolder(root=DATASET_DIR, transform=train_transforms)
        dataloader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

        class_names = dataset.classes
        with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
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
            log_str = f"Epoch {epoch+1}/{epochs} -> Độ lỗi (Loss): {epoch_loss:.4f} | Độ chính xác: {epoch_acc:.1f}%"
            logs.append(log_str)

        torch.save(model.state_dict(), MODEL_PATH)

        return jsonify({
            'message': 'Huấn luyện AI thành công!',
            'logs': logs
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi huấn luyện: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Sử dụng mô hình AI đã train để chẩn đoán bệnh từ ảnh"""
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Vui lòng chọn ảnh để chẩn đoán!'}), 400

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        return jsonify({'error': 'Chưa có mô hình AI! Vui lòng thêm dữ liệu và bấm Train AI trước.'}), 400

    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    model = build_model(len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
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