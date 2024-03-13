import common
import torch
from torchvision import models, transforms
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Define image preprocessing transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),  # Alter brightness and contrast
    transforms.RandomRotation(degrees=15),  # Random rotation of the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()  # Set model to evaluation mode

# Function to extract features from an image
def extracted_image_prop(link):
    try:
        # Check if the URL is in list format as a string and convert if necessary
        import ast
        if link.startswith("[") and link.endswith("]"):
            link = ast.literal_eval(link)[0]  # Safely evaluate string as list and get the first element

        response = requests.get(link)
        img = Image.open(BytesIO(response.content))
        img_t = image_transforms(img)
        img_t = img_t.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            properties = resnet(img_t)
        # Normalize extracted features
        properties = torch.nn.functional.normalize(properties, p=2, dim=1).squeeze().cpu().numpy()
        return properties 
    except requests.exceptions.RequestException as e:
        print(f"RequestException for URL {link}: {e}")
    except UnidentifiedImageError:
        print(f"UnidentifiedImageError: cannot identify image file from URL {link}.")
    except Exception as e:
        print(f"Unexpected error for URL {link}: {e}")
    return None


image_features = []
for index, row in common.df.iterrows():
    image_feature = extracted_image_prop(row['Image'])
    if image_feature is not None:
        image_features.append(image_feature)

# Save extracted features
common.os.makedirs(common.output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save extracted features
file_path = common.os.path.join(common.output_dir, 'image_features.pkl')
with open(file_path, 'wb') as f:
    pickle.dump(image_features, f)
