import os
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# CONFIG
# ==========================================
# Automatically use the GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"==============================")
print(f" Using device: {device}")
print(f"==============================")

DATA_DIR = "dataSet"
BATCH_SIZE = 4
EPOCHS = 3

# ==========================================
# TRANSFORMS
# ==========================================
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# CUSTOM DATASET
# ==========================================
class CustomLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        
        # 1. Manually define the classes and the dictionary mapping
        self.classes = ['blight', 'healthy']
        self.class_to_idx = {'blight': 0, 'healthy': 1}
        
        # 2. Manually gather all file paths and their matching labels
        self.image_paths = []
        self.labels = []
        
        # Find all blight images
        for path in glob.glob(f"{root_dir}/blight/*.jpg"):
            self.image_paths.append(path)
            self.labels.append(self.class_to_idx['blight'])
            
        # Find all healthy images
        for path in glob.glob(f"{root_dir}/healthy/*.jpg"):
            self.image_paths.append(path)
            self.labels.append(self.class_to_idx['healthy'])
            
        # 3. Create a 'samples' list of tuples to keep compatibility with Test 1
        self.samples = list(zip(self.image_paths, self.labels))

    def __len__(self):
        # PyTorch needs to know exactly how many total images exist
        return len(self.image_paths)

    def __getitem__(self, idx):
        # PyTorch asks for one specific image by its index number
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # Apply the ResNet transformations
        if self.transform:
            image = self.transform(image)

        # Return the transformed image tensor and its integer label
        return image, label

# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    dataset = CustomLeafDataset(root_dir=DATA_DIR, transform=resnet_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataset, dataloader

# ==========================================
# MODEL SETUP
# ==========================================
def build_model(num_classes):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# ==========================================
# TRAINING LOOP 
# ==========================================
def train(model, dataloader, epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f}")

# ==========================================
# PREDICTION
# ==========================================
def predict(model, img_tensor):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, idx = torch.max(output, 1)
    return idx.item()

def print_prediction(label):
    print("\n==============================")
    print(f"   AI PREDICTION → {label}")
    print("==============================")

# ==========================================
# VISUALIZATION
# ==========================================
def save_prediction_image(img, title, filename):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')

# ==========================================
# MAIN SCRIPT
# ==========================================
dataset, dataloader = load_data()
class_names = dataset.classes
model = build_model(len(class_names))
train(model, dataloader, EPOCHS)

# Test 1: Random image
idx = random.randint(0, len(dataset)-1)
path, true_label = dataset.samples[idx]
img = Image.open(path)

tensor = dataset[idx][0].unsqueeze(0).to(device)
pred_label = predict(model, tensor)

save_prediction_image(
    img,
    f"AI: {class_names[pred_label]} | Actual: {class_names[true_label]}",
    "Test1_folder.png"
)

# Test 2: Custom image
def test_custom(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return

    img = Image.open(path).convert("RGB")
    tensor = resnet_transform(img).unsqueeze(0).to(device)
    pred_label = predict(model, tensor)

    save_prediction_image(
        img,
        f"AI Prediction: {class_names[pred_label]}",
        "Test2_custom.png"
    )

test_custom("test_leaf.jpg")
print("Done.")