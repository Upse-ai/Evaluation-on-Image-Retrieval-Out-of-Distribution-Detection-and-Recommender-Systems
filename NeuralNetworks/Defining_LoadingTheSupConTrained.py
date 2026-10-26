import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# --- 1. MINIMAL WRAPPER ---
class SupConClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Here we are trying to create the model as we want that is to tak the features from second last layer 
        m = models.wide_resnet50_2()
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        # Now we add a classification layer as well, because the saved weights also store the weights of the classification layer, 
        # To get those, we add a layer at the end of feature extraction layer, so that when we load the weights on this, the model architecture must be matching.
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 1. Extracting raw features
        raw_feat = self.backbone(x).flatten(1)
        # 2. L2 Normalization (can be optional, but for getting logits from a contrastive model, it is mandatory)
        z = F.normalize(raw_feat, dim=1)
        # 3. Logits
        logits = self.fc(z)
        return raw_feat, logits

# --- 2. LOADING LOGIC ---
def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = len(ckpt['classes'])
    
    model = SupConClassifier(num_classes).to(device)
    # This is where we load the weights onto the model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['classes']

# --- 3. DUMMY EXTRACTION ---
if __name__ == "__main__":
    CKPT_PATH = "wrn50_supcon.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, class_names = load_model(CKPT_PATH, DEVICE)
        
        # Create a dummy image [B, C, H, W]
        dummy_img = torch.randn(1, 3, 224, 224).to(DEVICE)

        with torch.no_grad():
            raw_feat, logits = model(dummy_img)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)

        print(f"--- Extraction Results ---")
        print(f"Raw Feature (X) shape: {raw_feat.shape}")
        print(f"Logits (L) shape:      {logits.shape}")
        print(f"Confidence (C):        {conf.item():.4f}")
        print(f"Predicted Class (P):   {class_names[pred.item()]}")

    except FileNotFoundError:
        print("Checkpoint not found. Check your path!")