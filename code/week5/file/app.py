import gradio as gr
from predict_mood import predict_image

def predict(image):
    if image is None:
        return "Please upload an image."
    
    # Save the temporary image to pass to predict_image
    # Gradio passes the image as a numpy array or PIL Image depending on type
    # But predict_image expects a path.
    # Let's modify predict_image to accept PIL Image or path, OR just save it here.
    # Actually, let's modify predict_image to be more flexible, but for now saving is easier to keep predict_mood.py simple.
    
    # Wait, predict_mood.py loads the model every time. That's slow for an app.
    # I should load the model once globally in app.py.
    
    return predict_image_wrapper(image)

# Let's import the model loading parts to avoid reloading every time
import torch
from torchvision import transforms
from mood_pre import moodcnn
from PIL import Image

# Load model once
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = moodcnn()
model_path = 'mood_model.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASSES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def predict_image_wrapper(image):
    if model is None:
        return "Model not loaded."
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    try:
        # Gradio passes PIL Image by default if type="pil"
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = CLASSES[predicted.item()]
            return predicted_class
    except Exception as e:
        return f"Error: {e}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image_wrapper,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=1, label="Predicted Mood"),
    title="Mood Predictor",
    description="Upload a photo to detect the mood."
)

if __name__ == "__main__":
    iface.launch()
