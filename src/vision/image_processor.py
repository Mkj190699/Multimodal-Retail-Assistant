import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Processes images for retail product recognition and analysis.
    """
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize the vision processor with CLIP model.
        
        Args:
            model_name: CLIP model variant to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CLIP model {model_name} on {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Additional transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Product categories for retail
        self.product_categories = [
            "clothing", "electronics", "home decor", "books", 
            "groceries", "beauty products", "sports equipment",
            "furniture", "toys", "kitchen appliances"
        ]
        
    def extract_features(self, image_path: str) -> torch.Tensor:
        """
        Extract visual features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as torch tensor
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu()
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def detect_products(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect and classify products in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected products with details
        """
        # Load image with OpenCV for detection
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to PIL for CLIP
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare text inputs
        text_inputs = torch.cat([
            clip.tokenize(f"a photo of {cat}") for cat in self.product_categories
        ]).to(self.device)
        
        # Process image
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(3)
        
        # Return detected products
        detected_products = []
        for value, idx in zip(values, indices):
            detected_products.append({
                "category": self.product_categories[idx],
                "confidence": float(value),
                "bounding_box": self._estimate_bounding_box(image)  # Simplified
            })
        
        return detected_products
    
    def _estimate_bounding_box(self, image: np.ndarray) -> Dict[str, int]:
        """
        Estimate bounding box for detected object.
        In production, replace with proper object detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Bounding box coordinates
        """
        h, w = image.shape[:2]
        return {
            "x_min": int(w * 0.25),
            "y_min": int(h * 0.25),
            "x_max": int(w * 0.75),
            "y_max": int(h * 0.75),
            "width": int(w * 0.5),
            "height": int(h * 0.5)
        }
    
    def get_color_palette(self, image_path: str, n_colors: int = 5) -> List[str]:
        """
        Extract dominant colors from an image.
        
        Args:
            image_path: Path to the image file
            n_colors: Number of colors to extract
            
        Returns:
            List of hex color codes
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape image
        pixels = image.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)
        
        # Get colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert to hex
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        
        return hex_colors
    
    def get_textual_description(self, image_path: str) -> str:
        """
        Generate textual description of an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Textual description
        """
        # This would use a vision-language model in production
        # For now, return a simple description based on detected products
        products = self.detect_products(image_path)
        
        if not products:
            return "An image of various retail products"
        
        main_product = products[0]
        return f"An image showing {main_product['category']} with {main_product['confidence']:.0%} confidence"

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = VisionProcessor()
    
    # Example: Process an image
    try:
        # Replace with actual image path
        features = processor.extract_features("sample_product.jpg")
        products = processor.detect_products("sample_product.jpg")
        colors = processor.get_color_palette("sample_product.jpg")
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Detected products: {products}")
        print(f"Dominant colors: {colors}")
        
    except FileNotFoundError:
        print("Please provide a valid image file path")
