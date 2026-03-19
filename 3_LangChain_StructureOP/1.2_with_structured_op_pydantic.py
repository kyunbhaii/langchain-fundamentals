from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()


model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# schema
class Review(BaseModel):

    key_themes: list[str] = Field(description='Write down all the key themes discussed in the review in a list')
    summary: str = Field(description='A brief summary of the review')
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description='Return sentiment of the review either positive, negative, or neutral')
    pros: Optional[list[str]] = Field(default=None, description='Write down all the pros inside a list')
    cons: Optional[list[str]] = Field(default=None, description='Write down all the cons inside a list')
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review, method="json_mode")

result = structured_model.invoke(
    "Return valid JSON following this schema: "
    "key_themes (list of strings), summary (string), sentiment (positive, negative, or neutral), "
    "pros (list of strings or null), cons (list of strings or null). "
    "Review: I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! "
    "The Snapdragon 8 Gen 3 processor makes everything lightning fast whether I'm gaming, multitasking, or editing photos. "
    "The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver. "
    "The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. "
    "The 200MP camera is stunning, especially in night mode. "
    "However, the phone is bulky and heavy, includes bloatware, and is expensive."
)

print(result)
# print(type(result))
# print(result.summary)