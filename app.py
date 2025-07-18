
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io

# ✅ Define a CNN feature extractor
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 56 * 56, 512)  # Adjusted output size to 512 features

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ✅ Define a hybrid model combining ResNet50 and SimpleCNN
class LungCancerHybridModel(nn.Module):
    def __init__(self):
        super(LungCancerHybridModel, self).__init__()
        
        # ResNet50 as the main classifier
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove the last FC layer
        
        # CNN as a feature extractor
        self.cnn = SimpleCNN()
        
        # Combine CNN features with ResNet
        self.fc_combined = nn.Linear(2048 + 512, 3)  # Combined features (ResNet + CNN)

    def forward(self, x):
        cnn_features = self.cnn(x)  # Extract CNN features (512-d)
        resnet_features = self.resnet(x)  # Extract ResNet50 features (2048-d)
        combined_features = torch.cat((resnet_features, cnn_features), dim=1)  # (2048+512 = 2560)
        return self.fc_combined(combined_features)  # Pass through the final FC layer

# ✅ Initialize Flask app
app = Flask(__name__, template_folder=r"D:/Games/OneDrive/Desktop/pro 1/templates")

# ✅ Load the model
model = LungCancerHybridModel()

# ✅ Load saved model with proper handling
checkpoint = torch.load('lung_cancer_resnet_model.pth', map_location=torch.device('cpu'))

# 🔥 Remove the mismatched layer (but ideally, ensure checkpoint compatibility)
if 'resnet.conv1.weight' in checkpoint:
    del checkpoint['resnet.conv1.weight']

# ✅ Load remaining weights
model.load_state_dict(checkpoint, strict=False)

# 🔥 Freeze all ResNet layers except the last layer
for param in model.resnet.parameters():
    param.requires_grad = False
for param in model.resnet.fc.parameters():
    param.requires_grad = True

# ✅ Define optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ✅ Set model to evaluation mode
model.eval()

# ✅ Correct image preprocessing for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ✅ Keep normalization
])

# 🔥 Define class labels
CLASS_NAMES = ["No Cancer", "Low Risk", "High Risk"]

def get_cancer_stage(probability, predicted_class):
    if predicted_class == "No Cancer":
        return "No Cancer Detected"
    elif predicted_class == "Low Risk":
        return "Stage I - Early Detection" if probability < 50 else "Stage II - Moderate Cancer"
    elif predicted_class == "High Risk":
        if probability >= 85:
            return "Stage IV - Immediate Attention Required"
        elif probability >= 70:
            return "Stage III - Advanced Cancer"
        elif probability >= 50:
            return "Stage II - Moderate Cancer"
        else:
            return "Stage I - Early Detection"
    return "Unknown Stage"


# ✅ Home route
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # ✅ Convert image to RGB
        img = Image.open(io.BytesIO(file.read())).convert("RGB")  
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # ✅ Perform prediction
        model.eval()  # ✅ Ensure model is in evaluation mode
        with torch.no_grad():
            output = model(img)
            probabilities = torch.softmax(output, dim=1)[0]  
            predicted_class_idx = torch.argmax(output, dim=1).item()
            probability = round(probabilities[predicted_class_idx].item() * 100, 2)

            # ✅ Ensure `predicted_class` is assigned
            predicted_class = CLASS_NAMES[predicted_class_idx]

        # ✅ Get cancer stage
        cancer_stage = get_cancer_stage(probability, predicted_class)

        # ✅ Return JSON response
        return jsonify({
            "prediction": predicted_class,
            "probability": probability,
            "cancer_stage": cancer_stage
        })

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)