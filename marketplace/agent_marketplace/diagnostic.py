from sentence_transformers import SentenceTransformer
from app.routes.agents import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Sentence transformers package is available")
        logger.info(f"✓ Model loaded successfully: {model}")
    except Exception as e:
        logger.error(f"× Error loading sentence transformers: {str(e)}")

def check_routes():
    try:
        routes = [{"path": route.path, "name": route.name} for route in router.routes]
        logger.info("Registered routes:")
        for route in routes:
            logger.info(f"✓ {route['path']} ({route['name']})")
    except Exception as e:
        logger.error(f"× Error checking routes: {str(e)}")

if __name__ == "__main__":
    logger.info("Running diagnostics...")
    check_dependencies()
    check_routes()
