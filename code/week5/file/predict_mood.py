import torch
from torchvision import transforms
from PIL import Image
import sys
import os
from mood_pre import moodcnn

# Define the class names based on the directory structure
CLASSES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def predict_image(image_path, model_path='mood_model.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run mood_pre.py first to train the model.")
        return

    # Load the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = moodcnn()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.to(device)
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((48, 48)), # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    try:
        image = Image.open(image_path).convert('RGB') # Ensure 3 channels
        image = transform(image).unsqueeze(0).to(device) # Add batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = CLASSES[predicted.item()]
        return predicted_class

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_mood.py <image_path>")
        print("Example: python predict_mood.py cat.jpg")
    else:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        if result:
            print(f"Predicted Mood: {result}")
