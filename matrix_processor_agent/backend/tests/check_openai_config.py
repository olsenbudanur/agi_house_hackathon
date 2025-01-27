import asyncio
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_openai_config():
    """Check OpenAI client configuration."""
    client = OpenAI()
    
    # Check timeout settings
    logger.info(f"OpenAI client timeout: {getattr(client, 'timeout', 'Not set')}")
    logger.info(f"OpenAI client max_retries: {getattr(client, 'max_retries', 'Not set')}")
    
    # Check other relevant settings
    logger.info(f"OpenAI client base URL: {client.base_url}")
    logger.info(f"OpenAI client default headers: {client.default_headers}")

if __name__ == '__main__':
    asyncio.run(check_openai_config())
