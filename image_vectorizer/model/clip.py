import torch
from transformers import CLIPModel, CLIPProcessor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

repo = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(repo).to(DEVICE)
processor = CLIPProcessor.from_pretrained(repo)

model.save_pretrained('./clip-vit-base-patch32')
processor.save_pretrained('./clip-vit-base-patch32')