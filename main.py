from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import logging
import os
from contextlib import asynccontextmanager

from analyze import router as analyze_router
from compare import router as compare_router
from timeline import router as timeline_router
from visualize import router as visualize_router
from embedding_updater import router as updater_router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app_metadata = {
    "title": "adhoc - API",
    "description": """
    ## adhoc - API
    
    A comprehensive API for analyzing political text using advanced machine learning techniques.
    
    ### Features:
    - **Text Analysis**: Extract political ideology, sentiment, and key phrases from text
    - **Similarity Search**: Find similar political statements using FAISS vector search
    - **Multi-label Classification**: Classify text across social, economic, and foreign policy dimensions
    - **Attention Visualization**: See which words are most important for predictions
    - **Timeline Analysis**: Track ideological drift over time
    - **Text Comparison**: Compare two texts for ideological similarity
    - **Clustering**: Group texts by ideological similarity
    - **Visualization**: Interactive plots of political ideology space
    - **Database Management**: Add new political statements and update embeddings
    
    ### Endpoints:
    - `/analyze/` - Main analysis endpoint
    - `/compare/` - Compare two texts
    - `/timeline/{politician}/trajectory` - Political trajectory over time
    - `/visualize/` - Various visualization endpoints
    - `/update/` - Database management endpoints
    """,
    "version": "1.0.0",
    "contact": {"name": "adhoc", "email": "yooo@adhoc.com"},  # teehee
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Adhoc Political Analysis API...")

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/backups", exist_ok=True)

    try:
        from models import initialize_model

        initialize_model()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

    yield

    logger.info("Shutting down adhoc...")


app = FastAPI(lifespan=lifespan, **app_metadata)

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# health check models
class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    database_status: str


class AppInfo(BaseModel):
    name: str
    version: str
    description: str
    endpoints: dict
    features: list


# include routers with prefixes
app.include_router(
    analyze_router,
    prefix="/analyze",
    tags=["Analysis"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    compare_router,
    prefix="/compare",
    tags=["Comparison"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    timeline_router,
    prefix="/timeline",
    tags=["Timeline"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    visualize_router,
    prefix="/visualize",
    tags=["Visualization"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    updater_router,
    prefix="/update",
    tags=["Database Management"],
    responses={404: {"description": "Not found"}},
)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>adhoc API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .method { background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; margin-right: 10px; }
            .description { margin-top: 10px; color: #7f8c8d; }
            .feature { background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #27ae60; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .nav { text-align: center; margin: 20px 0; }
            .nav a { background: #3498db; color: white; padding: 10px 20px; margin: 0 10px; border-radius: 5px; text-decoration: none; }
            .nav a:hover { background: #2980b9; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>adhoc API</h1>
            
            <div class="nav">
                <a href="/docs">API Documentation</a>
                <a href="/redoc">ReDoc</a>
                <a href="/health">Health Check</a>
            </div>
            
            <h2>Key Features</h2>
            <div class="feature">Advanced political ideology classification using transformer models</div>
            <div class="feature">Semantic similarity search against politician statements</div>
            <div class="feature">Multi-dimensional analysis (social, economic, foreign policy)</div>
            <div class="feature">Attention-based token importance extraction</div>
            <div class="feature">Interactive visualizations with UMAP/t-SNE</div>
            <div class="feature">Real-time database updates and FAISS indexing</div>
            
            <h2>Main Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">POST</span><strong>/analyze/</strong>
                <div class="description">
                    Main analysis endpoint. Analyze political text for ideology, extract key phrases, 
                    and find similar statements from politicians.
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span><strong>/compare/</strong>
                <div class="description">
                    Compare two political texts for ideological similarity using cosine similarity 
                    and euclidean distance metrics.
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span><strong>/timeline/{politician}/trajectory</strong>
                <div class="description">
                    Analyze ideological evolution of a politician over time with PCA visualization.
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span><strong>/visualize/ideology-space</strong>
                <div class="description">
                    Create interactive UMAP visualization of political ideology space with your texts 
                    compared to reference politicians.
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span><strong>/visualize/cluster-analysis</strong>
                <div class="description">
                    Perform clustering analysis on political texts using K-means and visualize 
                    results with dimensionality reduction.
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span><strong>/update/add-statement</strong>
                <div class="description">
                    Add new political statements to the database with automatic embedding generation 
                    and FAISS index updates.
                </div>
            </div>
            
            <h2>Example Usage</h2>
            <p>Try analyzing this text: <em>"Healthcare is a fundamental right and we need universal coverage for all Americans"</em></p>
            
            <h2>🛠️ Technical Stack</h2>
            <ul>
                <li><strong>Backend:</strong> FastAPI, PyTorch, Transformers</li>
                <li><strong>ML:</strong> BERT-based ideology classifier with attention</li>
                <li><strong>Vector Search:</strong> FAISS for similarity search</li>
                <li><strong>Visualization:</strong> Plotly, UMAP, t-SNE</li>
                <li><strong>Data:</strong> Pandas, NumPy, scikit-learn</li>
            </ul>
            
            <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
                <p>Built for analyzing political discourse with state-of-the-art NLP</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        # Check if models are loaded
        from models import model

        models_loaded = model is not None

        # Check database status
        database_status = "healthy"
        try:
            from embedding_updater import updater

            stats = updater.get_stats()
            if stats.total_embeddings == 0:
                database_status = "empty"
        except Exception as e:
            database_status = f"error: {str(e)}"

        return HealthResponse(
            status="healthy",
            version=app_metadata["version"],
            models_loaded=models_loaded,
            database_status=database_status,
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version=app_metadata["version"],
            models_loaded=False,
            database_status=f"error: {str(e)}",
        )


@app.get("/info", response_model=AppInfo)
async def app_info():
    return AppInfo(
        name="Adhoc Political Analysis API",
        version=app_metadata["version"],
        description="Advanced political text analysis using machine learning",
        endpoints={
            "analyze": "Comprehensive political text analysis",
            "compare": "Compare two texts for ideological similarity",
            "timeline": "Track politician ideology over time",
            "visualize": "Interactive visualizations and clustering",
            "update": "Database management and embedding updates",
        },
        features=[
            "Political ideology classification",
            "Multi-label dimension analysis",
            "Attention-based token importance",
            "FAISS similarity search",
            "Interactive visualizations",
            "Real-time database updates",
            "Clustering analysis",
            "Timeline tracking",
        ],
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return {
        "error": "Endpoint not found",
        "available_endpoints": [
            "/docs",
            "/redoc",
            "/health",
            "/info",
            "/analyze/",
            "/compare/",
            "/timeline/",
            "/visualize/",
            "/update/",
        ],
    }


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "message": "Please check the logs"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
