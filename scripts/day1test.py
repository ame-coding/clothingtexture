import torch
from PIL import Image
from torchvision import transforms

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
img = Image.open("data/test.jpg")

# Apply transform
img_tensor = transform(img)

# Add batch dimension
img_tensor = img_tensor.unsqueeze(0)

print(img_tensor.shape)