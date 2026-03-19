from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite"
)

# Chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Load chat history from file
chat_history = []
with open("chat_history.txt", "r") as f:
    for line in f:
        # each line should be a serialized message object
        chat_history.append(line.strip())

# print("Loaded chat history:")
# print(chat_history)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower().strip() in ["exit", "quit"]:
        print("Ending chat.")
        break

    # Invoke the prompt
    prompt = chat_template.invoke({
        "chat_history": chat_history,
        "query": user_input
    })

    # Send to model
    response = model.invoke(prompt)

    # Extract the text
    answer = response.output_text
    print("AI:", answer)

    # Append to history as simple text
    chat_history.append(f"User: {user_input}")
    chat_history.append(f"AI: {answer}")

    # Optionally save back to file
    with open("chat_history.txt", "a") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"AI: {answer}\n")