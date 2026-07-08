import os
import copy
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from functools import lru_cache
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# ==========================================
# CONFIG & PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ==========================================
# 1. DATASET CLASS
# ==========================================
class CustomLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.classes = ['blight', 'healthy']
        self.class_to_idx = {'blight': 0, 'healthy': 1}
        
        self.image_paths = []
        self.labels = []
        
        for path in glob.glob(f"{root_dir}/blight/*.jpg"):
            self.image_paths.append(path)
            self.labels.append(self.class_to_idx['blight'])
            
        for path in glob.glob(f"{root_dir}/healthy/*.jpg"):
            self.image_paths.append(path)
            self.labels.append(self.class_to_idx['healthy'])
            
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
# 2. VISION MODEL CLASS
# ==========================================
class BlightVisionModel:
    def __init__(self, data_dir, batch_size=8, seed=42):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Device configuration
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"[VisionModel] Using device: {self.device}")

        # Seeding
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Transforms
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load Data and build model
        self.train_loader, self.val_loader, self.dataset = self._prepare_data()
        self.class_names = self.dataset.classes
        self.model = self._build_model(len(self.class_names))

    def _prepare_data(self):
        train_dataset = CustomLeafDataset(root_dir=self.data_dir, transform=self.train_transform)
        eval_dataset = CustomLeafDataset(root_dir=self.data_dir, transform=self.eval_transform)

        num_samples = len(eval_dataset)
        indices = list(range(num_samples))
        random.shuffle(indices)
        split = max(1, int(0.2 * num_samples))
        val_idx, train_idx = indices[:split], indices[split:]

        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(Subset(eval_dataset, val_idx), batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        return train_loader, val_loader, eval_dataset

    def _build_model(self, num_classes):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def train(self, epochs=25, save_path=None):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam([
            {"params": self.model.layer4.parameters(), "lr": 1e-4},
            {"params": self.model.fc.parameters(), "lr": 1e-3},
        ])

        best_val_acc = 0.0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(self.train_loader):.4f} | Val Acc: {val_acc*100:.1f}%")

            # Keep the weights from the best-performing epoch
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                if save_path:
                    torch.save(best_state, save_path)
                    print(f"  ↳ New best model saved (Val Acc: {val_acc*100:.1f}%)")

        # Restore the best checkpoint before returning
        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\nBest validation accuracy: {best_val_acc*100:.1f}%")
        return best_val_acc

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"[VisionModel] Loaded weights from {model_path}")

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / max(1, total)

    def predict_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.eval_transform(img).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).squeeze(0)
            conf, idx = torch.max(probs, 0)

        label = self.class_names[idx.item()]
        class_probs = {name: probs[i].item() for i, name in enumerate(self.class_names)}
        return label, conf.item(), class_probs, img

    @staticmethod
    def save_prediction(img, title, filename):
        plt.figure()
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

# ==========================================
# 3. RAG AGENT CLASS
# ==========================================
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")

class BlightRAGAgent:
    def __init__(self):
        self.response_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.grader_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.blight_knowledge = [
            "Apply copper-based fungicides immediately to prevent the blight spores from spreading.",
            "Remove and burn all infected foliage. Do not compost infected leaves as spores will survive.",
            "Ensure proper spacing between potato plants to improve airflow and reduce leaf wetness.",
            "Practice crop rotation; avoid planting potatoes or tomatoes in the same soil for 3-4 years.",
            "Water plants at the base early in the day so the sun can dry the foliage quickly.",
            "Scout fields regularly during warm, humid weather when late blight develops fastest.",
            "Use certified disease-free seed potatoes and resistant cultivars to lower the risk."
        ]
        
        self.retriever = self._build_retriever()
        
        # Tools need to be defined safely for LangGraph
        @tool
        def retrieve_blight_strategies(query: str) -> str:
            """Search and return mitigation and treatment strategies for potato/tomato blight."""
            docs = self.retriever.invoke(query)
            return "\n\n".join(doc.page_content for doc in docs)
            
        self.tools = [retrieve_blight_strategies]
        self.graph = self._build_graph()
        print("[RAGAgent] Agentic RAG graph ready.")

    def _build_retriever(self):
        documents = [Document(page_content=text) for text in self.blight_knowledge]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = InMemoryVectorStore.from_documents(documents, embedding=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    def _generate_query_or_respond(self, state: MessagesState):
        response = self.response_model.bind_tools(self.tools).invoke(state["messages"])
        return {"messages": [response]}

    def _grade_documents(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = f"Grade relevance of context to question. Context: {context}\nQuestion: {question}. Answer 'yes' or 'no'."
        response = self.grader_model.with_structured_output(GradeDocuments).invoke([{"role": "user", "content": prompt}])
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

    def _rewrite_question(self, state: MessagesState):
        question = state["messages"][0].content
        prompt = f"Look at the input and formulate an improved question: {question}"
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    def _generate_answer(self, state: MessagesState):
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = f"Answer concisely based on context.\nQuestion: {question}\nContext:\n{context}"
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    def _route_on_tool_calls(self, state: MessagesState):
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    def _build_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            self._route_on_tool_calls,
            {"tools": "retrieve", END: END}
        )
        workflow.add_conditional_edges("retrieve", self._grade_documents)
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        return workflow.compile()

    def get_advice(self, query: str) -> str:
        result = self.graph.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content


# ==========================================
# 4. MAIN ORCHESTRATION
# ==========================================
def main():
    # Set RETRAIN = True to train from scratch on the current dataset,
    # ignoring (and overwriting) any saved checkpoint. Use this whenever
    # you add or change images in dataSet/.
    RETRAIN = False

    data_dir = os.path.join(PROJECT_ROOT, "dataSet")
    test_image_path = os.path.join(PROJECT_ROOT, "test_leaf.jpg")
    model_path = os.path.join(SCRIPT_DIR, "blight_model.pth")

    # 1. Initialize Vision Model, then load a saved checkpoint or train a new one
    vision_system = BlightVisionModel(data_dir=data_dir, batch_size=8)

    if os.path.exists(model_path) and not RETRAIN:
        vision_system.load(model_path)
        val_acc = vision_system.evaluate()
        print(f"\nValidation accuracy (loaded model): {val_acc*100:.1f}%")
    else:
        if RETRAIN:
            print("[RETRAIN] Training from scratch on current dataset...")
        vision_system.train(epochs=25, save_path=model_path)
        val_acc = vision_system.evaluate()
        print(f"\nValidation accuracy: {val_acc*100:.1f}%")

    # 2. Test Custom Image
    if not os.path.exists(test_image_path):
        print(f"Missing test file: {test_image_path}")
        return

    print(f"\n[RUNNING TEST: Custom Image -> {test_image_path}]")
    prediction_name, confidence, class_probs, img = vision_system.predict_image(test_image_path)
    print(f"AI PREDICTION → {prediction_name} ({confidence*100:.1f}% confidence)")
    print("  Class probabilities: " + " | ".join(f"{name}: {p*100:.1f}%" for name, p in class_probs.items()))

    vision_system.save_prediction(
        img,
        f"AI Prediction: {prediction_name} ({confidence*100:.1f}%)",
        "test_leaf_result.png"
    )

    # 3. Consult Agent if Blight Detected
    if prediction_name == 'blight':
        print("\n[ALERT] Consulting agentic RAG for mitigation strategies...")
        rag_agent = BlightRAGAgent()
        answer = rag_agent.get_advice(
            "The crop has been diagnosed with blight. How do I treat it and stop the spread?"
        )
        print("\n[AGENTIC RAG RESPONSE]")
        print(answer)

if __name__ == '__main__':
    main()