from requests import get
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from db import VectorDB

import configparser


config = configparser.ConfigParser()
config.read('config.INI')

# Accessing configuration values
api_key = config.get('General', 'GEMINI_API_KEY')

# Configure and Initialise Gemini
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel('gemini-pro')

vectordb = VectorDB()
resp = vectordb.create_and_update_vectordb()

# Tools
# Email Writer
def find_receiver(text):
    if "hr" in text.lower():
        return "hr@arjtech.com"

    if "manager" in text.lower():
        return "manager@arjtech.com"

    return "xyz@arjtech.com"

def write_an_email(about):

    try:

        sender = "Arjun V"
        receiver = find_receiver(about)
        length = 150

        prompt = '''You are an email wirting assistant at the company ArjTech Pvt Ltd. 
        Employees use you for writing professional emails to different people within the company.
        Write an email from {} to {}. The email is about {}. The email should be formal and {} characters long.
        Only use the information provided in the context, never hallucinate.
        '''.format(sender, receiver, about, length)
        
        response = gemini.generate_content(prompt)

    except Exception as e:
        return None, f"An error occurred while formatting the email :: {e}"

    return response.text, f"An email regarding this matter has been sent to the {receiver}. Please wait for their response."


# Query Resolver
def make_rag_prompt(query, relevant_passage):

    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"You are a helpful and informative chatbot at ArjTech Pvt Ltd that answers questions using text from the reference passage included below. "
        f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
        f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"Only include the details given in the passage, never hallucinate."
        f"QUESTION: \'{query}\'\n"
        f"PASSAGE: \'{relevant_passage}\'\n\n"
        f"ANSWER:"
    )
    return prompt

def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text

def answer_query(query):

    relevant_text = vectordb.get_relevant_docs(query)
    text = " ".join(relevant_text)

    prompt = make_rag_prompt(query, relevant_passage=text)
    answer = generate_response(prompt)

    return answer, "True"

# create a note
def create_mom(description):

    file_path="notes.pdf"

    prompt = '''You are a writing assistant at Arjtech Pvt Ltd, summarize the following meeting discussion 
    into organized minutes. Ensure to capture all important details 
    while ignoring any irrelevant context. The output should include sections 
    for attendees, agenda items, key discussions, decisions made, and action items. 
    Do not include any fabricated information. Here is the meeting description: {}
    '''.format(description)

    response = gemini.generate_content(prompt)

    with open(file_path, "w") as file:
        file.write(response.text)

    return response.text, f"File {file_path} created successfully."
