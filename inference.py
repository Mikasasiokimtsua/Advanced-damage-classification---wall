CLASS2NUM = {'Class A': 18, 'Class B': 19, 'Class C': 20}
CRITERIA_GROUPS = {
    18: [0, 2],
    19: [3, 4, 6, 8],
    20: [1, 5, 7, 9, 10]
}

CRITERIA_NAMES = [
    "Exposed rebar",
    "No significant damage",
    "Huge Spalling",
    "X and V-shaped cracks",
    "Continuous Diagonal cracks",
    "Discontinuous Diagonal cracks",
    "Continuous vertical cracks",
    "Discontinuous vertical cracks",
    "Continuous horizontal cracks",
    "Discontinuous horizontal cracks",
    "Small cracks"
]


import os
import torch
from torchvision import transforms, models
from PIL import Image
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

from torchvision.datasets import ImageFolder
damage_class_names = ['Class A', 'Class B', 'Class C']

damage_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
damage_model.classifier[2] = torch.nn.Linear(damage_model.classifier[2].in_features, 3)
damage_model.load_state_dict(torch.load("best_damage_model.pth"))
damage_model.eval().to(device)

crack_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
crack_model.classifier[2] = torch.nn.Linear(crack_model.classifier[2].in_features, 11)
crack_model.load_state_dict(torch.load("best_crack_model.pth"))
crack_model.eval().to(device)

CLASS2NUM = {'Class A': 18, 'Class B': 19, 'Class C': 20}
CRITERIA_GROUPS = {
    18: [0, 2],
    19: [3, 4, 6, 8],
    20: [1, 5, 7, 9, 10]
}
CRITERIA_NAMES = [
    "Exposed rebar", "No significant damage", "Huge Spalling",
    "X and V-shaped cracks", "Continuous Diagonal cracks", "Discontinuous Diagonal cracks",
    "Continuous vertical cracks", "Discontinuous vertical cracks",
    "Continuous horizontal cracks", "Discontinuous horizontal cracks", "Small cracks"
]

test_dir = "datasets/test_data/wall"
test_imgs = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')],
                   key=lambda x: int(os.path.splitext(x)[0]))

results = []

for img_name in test_imgs:
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img_tensor = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = damage_model(img_tensor)
        pred_class_idx = out.argmax(1).item()
        class_name = damage_class_names[pred_class_idx]
        damage_class = CLASS2NUM[class_name]

    with torch.no_grad():
        crack_out = crack_model(img_tensor)
        probs = torch.sigmoid(crack_out).cpu().numpy()[0]
    allowed = CRITERIA_GROUPS[damage_class]
    crack_pred = [i for i, p in enumerate(probs) if i in allowed and p >= 0.5]
    if not crack_pred:
        allowed_probs = [(i, probs[i]) for i in allowed]
        max_idx = max(allowed_probs, key=lambda x: x[1])[0]
        crack_pred = [max_idx]

    value = ",".join([str(damage_class)] + [str(i) for i in crack_pred])
    row = [os.path.splitext(img_name)[0], value]
    results.append(row)

with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "class"])
    writer.writerows(results)
print("Submission saved to submission.csv")
