from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import base64
import io
import pickle

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_clip = model_clip.to(device)

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    data = pickle.load(f)
    le_master = data["le_master"]
    le_sub = data["le_sub"]
    price_min = data["price_min"]
    price_max = data["price_max"]

category_texts = list(le_sub.classes_)

class CrossAttentionFusionModel(nn.Module):
    def __init__(self, clip_model, category_texts, processor, num_master, num_sub, device):
        super().__init__()
        self.clip = clip_model
        self.device = device
        for param in self.clip.parameters():
            param.requires_grad = False
        with torch.no_grad():
            text_inputs = processor(text=category_texts, return_tensors="pt", padding=True).to(device)
            self.register_buffer('text_embeds', clip_model.get_text_features(**text_inputs))
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.master_head = nn.Linear(256, num_master)
        self.sub_head = nn.Linear(256, num_sub)
        self.price_head = nn.Linear(256, 1)

    def forward(self, pixel_values, return_attention=False):
        img_features = self.clip.get_image_features(pixel_values=pixel_values)
        query = img_features.unsqueeze(1)
        kv = self.text_embeds.unsqueeze(0).expand(query.size(0), -1, -1)
        attn_out, attn_weights = self.attention(query, kv, kv)
        x = self.fusion(attn_out.squeeze(1))
        master_out = self.master_head(x)
        sub_out = self.sub_head(x)
        price_out = self.price_head(x).squeeze(1)
        if return_attention:
            return master_out, sub_out, price_out, attn_weights
        return master_out, sub_out, price_out

# Load model
model = CrossAttentionFusionModel(
    model_clip, category_texts, processor,
    len(le_master.classes_), len(le_sub.classes_), device
).to(device)
model.load_state_dict(torch.load("fashion_model_ca.pth", map_location=device))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img_data = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(img_data)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        master_out, sub_out, price_out, attn_weights = model(
            inputs["pixel_values"], return_attention=True
        )

    pred_master = le_master.classes_[master_out.argmax(1).item()]
    pred_sub = le_sub.classes_[sub_out.argmax(1).item()]
    pred_price = price_out.item() * (price_max - price_min) + price_min

    attn = attn_weights.squeeze().tolist()
    top5_idx = sorted(range(len(attn)), key=lambda i: attn[i], reverse=True)[:5]
    top5 = [{"category": le_sub.classes_[i], "score": round(attn[i], 4)} for i in top5_idx]

    return jsonify({
        "masterCategory": pred_master,
        "subCategory": pred_sub,
        "price": round(pred_price, 2),
        "attention": top5
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)