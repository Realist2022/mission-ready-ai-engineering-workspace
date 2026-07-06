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

# New imports for the RAG / Mitigation Database
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIG
# ==========================================
# Automatically use GPU if available, otherwise fallback to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") # For Apple Silicon
else:
    device = torch.device("cpu")

print(f"==============================")
print(f"Using device: {device}")
print(f"==============================")

DATA_DIR = "dataSet"
BATCH_SIZE = 16
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
            
        # 3. Create a 'samples' list of tuples
        self.samples = list(zip(self.image_paths, self.labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    dataset = CustomLeafDataset(root_dir=DATA_DIR, transform=resnet_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True 
    )
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
    plt.close() # Close plot to free memory

# ==========================================
# RAG / MITIGATION DATABASE
# ==========================================
def setup_mitigation_db():
    print("Initializing ChromaDB and Sentence Transformer...")
    # Initialize ChromaDB client (in-memory)
    chroma_client = chromadb.Client()
    
    # Create a collection to hold our text data
    collection = chroma_client.get_or_create_collection(name="blight_strategies")
    
    # Load a lightweight, fast sentence embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Agricultural knowledge base
    documents = [
        "Apply copper-based fungicides immediately to prevent the blight spores from spreading.",
        "Remove and burn all infected foliage. Do not compost infected leaves as spores will survive.",
        "Ensure proper spacing between potato plants to improve airflow and reduce leaf wetness.",
        "Practice crop rotation; avoid planting potatoes or tomatoes in the same soil for 3-4 years.",
        "Water plants at the base early in the day so the sun can dry the foliage quickly."
    ]
    
    # Generate embeddings for the documents
    embeddings = embedder.encode(documents).tolist()
    ids = [f"strat_{i}" for i in range(len(documents))]
    
    # Load them into Chroma
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=ids
    )
    
    return collection, embedder

def get_mitigation_suggestions(collection, embedder, query, top_k=2):
    query_embedding = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return results['documents'][0]

# ==========================================
# MAIN SCRIPT
# ==========================================
if __name__ == '__main__':
    # 1. Setup Vision Model
    dataset, dataloader = load_data()
    class_names = dataset.classes
    model = build_model(len(class_names))
    train(model, dataloader, EPOCHS)

    # 2. Setup Vector Database
    db_collection, embedder = setup_mitigation_db()

    # ==========================================
    # TEST 1: Random Image from Dataset
    # ==========================================
    print("\n[RUNNING TEST 1: Random Dataset Image]")
    idx = random.randint(0, len(dataset)-1)
    path, true_label = dataset.samples[idx]
    img = Image.open(path)

    tensor = dataset[idx][0].unsqueeze(0).to(device)
    pred_label = predict(model, tensor)

    save_prediction_image(
        img,
        f"AI: {class_names[pred_label]} | Actual: {class_names[true_label]}",
        "random_test_result_folder.png"
    )
    print(f"Test 1 Saved as 'random_test_result_folder.png'")

    # ==========================================
    # TEST 2: Custom Image with ChromaDB RAG
    # ==========================================
    def test_custom_with_suggestions(path):
        print(f"\n[RUNNING TEST 2: Custom Image -> {path}]")
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            return

        img = Image.open(path).convert("RGB")
        tensor = resnet_transform(img).unsqueeze(0).to(device)
        pred_label = predict(model, tensor)
        prediction_name = class_names[pred_label]
        
        print_prediction(prediction_name)
        
        # If the AI detects blight, trigger the ChromaDB query
        if prediction_name == 'blight':
            print("\n[ALERT] Fetching mitigation strategies from database...")
            suggestions = get_mitigation_suggestions(
                db_collection, 
                embedder, 
                query="How do I treat and stop the spread of potato blight?", 
                top_k=2
            )
            
            for i, suggestion in enumerate(suggestions, 1):
                print(f"Suggestion {i}: {suggestion}")
                
        save_prediction_image(
            img,
            f"AI Prediction: {prediction_name}",
            "test_leaf_result_custom.png"
        )
        print(f"Test 2 Saved as 'test_leaf_result_custom.png'")

    test_custom_with_suggestions("test_leaf.jpg")
    print("\nDone.")