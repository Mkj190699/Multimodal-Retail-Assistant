"""
FastAPI application for Multimodal Retail Assistant.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from datetime import datetime

# Import your modules
from src.vision.image_processor import VisionProcessor
from src.nlp.query_processor import QueryProcessor
from src.rag.retriever import VectorRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Retail Assistant API",
    description="AI-powered retail assistant for personalized shopping recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vision_processor = VisionProcessor()
query_processor = QueryProcessor()
retriever = VectorRetriever()

# Request/Response models
class ProductRequest(BaseModel):
    image_url: Optional[str] = None
    text_query: str
    user_id: Optional[str] = None
    filters: Optional[dict] = None

class ProductResponse(BaseModel):
    recommendations: List[dict]
    confidence: float
    reasoning: Optional[str] = None
    processing_time: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_versions: dict

# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Multimodal Retail Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_versions={
            "vision": "CLIP-ViT-B/32",
            "nlp": "GPT-4",
            "retriever": "chroma-v1.0"
        }
    )

@app.post("/process", response_model=ProductResponse)
async def process_request(request: ProductRequest):
    """
    Process multimodal request and return recommendations.
    
    Args:
        request: ProductRequest containing image URL, text query, etc.
    
    Returns:
        Product recommendations with confidence scores
    """
    start_time = datetime.now()
    
    try:
        # Process image if provided
        image_features = None
        if request.image_url:
            logger.info(f"Processing image: {request.image_url}")
            image_features = vision_processor.extract_features(request.image_url)
        
        # Process text query
        logger.info(f"Processing text query: {request.text_query}")
        query_embedding = query_processor.encode_query(request.text_query)
        
        # Combine features (multimodal fusion)
        combined_features = None
        if image_features is not None:
            # Simple concatenation - can be improved with attention
            combined_features = torch.cat([query_embedding, image_features.flatten()])
        else:
            combined_features = query_embedding
        
        # Retrieve similar products
        logger.info("Retrieving similar products...")
        similar_products = retriever.search(
            query_vector=combined_features.numpy(),
            filters=request.filters,
            k=10
        )
        
        # Generate recommendations
        recommendations = []
        for product in similar_products:
            recommendations.append({
                "product_id": product["id"],
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "similarity_score": float(product["similarity"]),
                "image_url": product["image_url"],
                "description": product.get("description", "")
            })
        
        # Calculate average confidence
        avg_confidence = sum(p["similarity_score"] for p in recommendations) / len(recommendations)
        
        # Generate reasoning
        reasoning = query_processor.generate_reasoning(
            query=request.text_query,
            products=recommendations[:3]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ProductResponse(
            recommendations=recommendations,
            confidence=avg_confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload image for processing."""
    try:
        # Save uploaded file
        contents = await file.read()
        file_path = f"uploads/{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Process image
        detected_products = vision_processor.detect_products(file_path)
        colors = vision_processor.get_color_palette(file_path)
        description = vision_processor.get_textual_description(file_path)
        
        return {
            "filename": file.filename,
            "detected_products": detected_products,
            "dominant_colors": colors,
            "description": description,
            "file_size": len(contents)
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/{product_id}")
async def get_product_details(product_id: str):
    """Get details for a specific product."""
    # This would query your database
    return {
        "product_id": product_id,
        "name": "Sample Product",
        "price": 99.99,
        "category": "electronics",
        "description": "A sample product description",
        "features": ["feature1", "feature2"],
        "rating": 4.5,
        "review_count": 123
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
