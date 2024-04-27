# ragBot
# RagBot

RagBot is an AI-powered chatbot that provides information and assistance to users based on a predefined context. It uses Streamlit for the user interface, Langchain for document processing and retrieval, and Hugging Face Transformers for sentence embeddings.

## Features

- Answers user questions based on predefined question-answer pairs
- Handles out-of-scope questions by generating responses using LLMs and context from a JSON file
- Utilizes sentence embeddings to find the most relevant question-answer pair
- Provides a user-friendly interface using Streamlit

## Prerequisites

Before running RagBot, ensure that you have the following:

- Python 3.7 or higher
- An OpenAI API key

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/codexnyctis/ragBot
   ```

2. Navigate to the project directory:

   ```
   cd ragBot
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project directory and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your-api-key
   ```

## Usage

1. Prepare your data/OR use the example data:
   - Create a `data.json` file in the project directory.
   - The file should contain an array of objects, where each object represents a question-answer pair in the following format:
     ```json
     [
       {
         "question": "What is RagBot?",
         "answer": "RagBot is an AI-powered chatbot that provides information and assistance based on predefined context."
       },
       ...
     ]
     ```

2. Run the Streamlit app:

   ```
   streamlit run ragBot.py
   ```

3. Open the provided URL in your web browser.

4. Enter your question in the chat input field and press Enter.

5. RagBot will provide a response based on the provided context:
   - If the question is similar to one of the predefined questions in `data.json`, the corresponding answer will be displayed.
   - If the question is out-of-scope (i.e., not similar to any predefined question), RagBot will generate a response using the LLMs and the context from `data.json`.

## Customization

- To modify RagBot's behavior or add more functionality, you can update the code in `ragbot.py`.
- To change the predefined questions and answers, modify the `data.json` file.
- You can customize the LLMs and models used by modifying the relevant code sections in `ragbot.py`.

