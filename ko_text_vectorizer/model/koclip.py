import torch
from transformers import AutoModel, AutoProcessor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

repo = "Bingsu/clip-vit-base-patch32-ko"
model = AutoModel.from_pretrained(repo).to(DEVICE)
processor = AutoProcessor.from_pretrained(repo)

model.save_pretrained('./clip-vit-base-patch32-ko')
processor.save_pretrained('./clip-vit-base-patch32-ko')