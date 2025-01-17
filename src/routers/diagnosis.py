from fastapi import APIRouter
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import os

# Load environment variables from .env file
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY")
# )

router = APIRouter()

@router.post("/diagnosis")
async def diagnose_image(prediction):
    """
    Accepts the results of the ML Prediction on diseases and passes to llm call then returns the result for display on the front-end
    """
    if prediction == "healthy_hive":
        return
    
    prompt = f"A beehive photo was found to contain {prediction}.  What steps should the beekeeper take?  Include any recommendations regarding contacting local organizations as required by law as well as common remedies."

    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Access the message content correctly
    message_content = completion.choices[0].message.content  # Use `.content` attribute
    # print(message_content)
    return {"message": message_content}
