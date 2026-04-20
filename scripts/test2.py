import torch
from PIL import Image
from torchvision import transforms, models

# STEP 1: preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("data/test.jpg")
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)

# STEP 2: load VGG16
model = models.vgg16(pretrained=True)
model = model.features
model.eval()

# STEP 3: pass through model
with torch.no_grad():
    output = model(img_tensor)

print("Feature map shape:", output.shape)

# STEP 4: flatten
features = output.view(output.size(0), -1)
print("Feature vector shape:", features.shape)