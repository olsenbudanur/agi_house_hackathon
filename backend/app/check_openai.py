import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())  # This will search parent directories for .env file

def check_configuration():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"OpenAI API Key exists: {bool(api_key)}")
        print(f"OpenAI API Key length: {len(api_key) if api_key else 0}")
        print(f"API Key prefix: {api_key[:10]}... (showing first 10 chars)")
        
        # Test OpenAI client initialization
        client = OpenAI(api_key=api_key)
        
        # List all available models first
        print("\nListing all available models:")
        models = client.models.list()
        for model in models.data:
            print(f"- {model.id}")
            
        # Try a simple completion
        print("\nTesting basic completion API access...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("Basic completion test successful!")
        print(f"Response: {response.choices[0].message.content}")
        
        # Test vision API specifically
        print("\nTesting vision API access...")
        # Create a simple test image (1x1 pixel base64 PNG)
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        vision_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "url": f"data:image/png;base64,{test_image}"
                        },
                        {
                            "type": "text",
                            "text": "What's in this image?"
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        print("Vision API test successful!")
        print(f"Vision response: {vision_response.choices[0].message.content}")
        
    except Exception as e:
        print(f"\nError during testing:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")

if __name__ == "__main__":
    check_configuration()
