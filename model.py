"""
web app chatbot Gemini function calling
"""
import os
from dotenv import load_dotenv
import urllib
import urllib.request
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

import vertexai
from vertexai import generative_models
from google.cloud import bigquery
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    GenerationConfig,
    Tool,
    Part,
    SafetySetting
)

from toolbox import create_mom, write_an_email, answer_query


#AUTHIENTICATION
if "auth" not in st.session_state:
    load_dotenv()
    st.session_state.PROJECT_ID = os.getenv('PROJECT_ID')
    st.session_state.LOCATION = os.getenv('LOCATION')
    vertexai.init(project=st.session_state.PROJECT_ID , location=st.session_state.LOCATION)
    st.session_state.bq_client = bigquery.Client(project=st.session_state.PROJECT_ID)
    st.session_state.auth = True

# Functions declaration
create_mom_yaml = FunctionDeclaration(
    name="create_mom",
    description="This function processes spoken language descriptions of a meeting and organizes them into structured minutes, ensuring all important context is retained while irrelevant details are ignored.",
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "A text input containing the conversational description of the meeting, including discussions, decisions, and action items.",
            }
        },
    },
)


write_an_email_yaml = FunctionDeclaration(
    name="write_an_email",
    description="This function generates a formal email template by taking information about the email content as input.",
    parameters={
        "type": "object",
        "properties": {
            
            "about": {
                "type": "string",
                "description": "Text input specifying the content of the email.",
            },
        },
    },
)


answer_query_yaml = FunctionDeclaration(
    name="answer_query",
    description="A function for retrieving documents and generating answers based on employee queries about company policies.",
    parameters={
        "type": "object",
        "properties": {

            "query": {
                "type": "string",
                "description": "the main topic on which an employee seeks clarification, which could include leave policies, dress codes, perks, and other relevant company policies.",
            }
        },
    },
)

query_tools = Tool(
    function_declarations=[create_mom_yaml,
                        write_an_email_yaml,
                        answer_query_yaml],
)

#WEB App Interface
st.set_page_config(
    page_title="AI Agent - Employee Assistant",
    page_icon=":male-office-worker:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("AI Agent - Employee Assistant")

tile = st.container(border=True)
tile.write(
"""
This is the AI Gemini Agent at Arjtech Pvt Ltd, designed to facilitate seamless 
communication within the organization. Employees can easily contact anyone in the 
company, inquire about company policies through a chat interface, and make moms of meeting during 
conversations. All notes can be saved as PDF documents for easy reference.
""")

tile = st.container(border=True)
tile.markdown("#### Sample Prompts")
col1, col2 = tile.columns(2)

with col1.expander("Writing Emails", expanded=False):
    st.write(
        """
        - Send an email to the HR department asking for details about the benefits package.

        - Draft an email to the manager asking for 2 days wfh in october?

        - Draft an email to HR requesting leave from 25-11-2024 to 27-11-2024 for attending a family function?
    
        - Write an email to HR asking for details about the health insurance plan?
    """
)

with col2.expander("Answering Queries", expanded=False):
    st.write(
        """
        - Can you summarize the company leave policy?

        - What is the policy regarding taking vacation or sick leave at Arjtech?

        - Can you summarize the company health and safety policies?
    
        - What are the main rules in our employee dress code?
    """
)

with st.expander("Prepare MOMs", expanded=False):
    st.write(
        """
        - In todays product development progress meeting, Mark provided an update on the development 
        of core features, confirming that all primary functionalities had been successfully implemented. 
        However, a few minor bugs were identified, which he committed to resolving by Wednesday. Jane shared 
        feedback from initial user testing, noting that users had issues with the navigation, which required 
        some adjustments. She agreed to deliver updated mockups by Friday for the team to review. Emma discussed 
        the testing schedule, stressing the need to initiate integration tests next week. She will finalize the test 
        plan and share it with the team by Monday.

        - Please take the following meeting description and generate organized meeting minutes:
        "In todays meeting, we reviewed the progress of the current project. 
        Alice shared the updates on the design phase and mentioned that the team is on track. 
        Bob raised concerns about resource allocation and proposed a plan to hire two additional developers. 
        The team agreed to finalize the project timeline by next week, and Sarah volunteered 
        to draft the timeline document. Action items include: Alice will provide the design mockups 
        by Friday, and Bob will prepare a resource allocation report by Tuesday."
    """
)

#keep sessions
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'gemini_history' not in st.session_state:
    st.session_state.gemini_history = []

st.session_state.model = GenerativeModel(
                            model_name="gemini-1.0-pro-002",

                            system_instruction=["You are an AI Assistant developed for Arjtech. Your primary tasks are to write emails, retrieve information about company policies and assist in organizing meeting minutes using the provided tools."
                            , "You will process spoken language descriptions to generate structured meeting summaries."
                            , "Do not respond to questions outside these tasks.!"
                            , "Please use the tools provided to give concise answers"
                            , "DO NOT MAKEUP UP ANY ANSWERS IF NOT PROVIDED BY THE TOOLS!"],
                            
                            generation_config=GenerationConfig(temperature=0,
                                                                top_p=0.95,
                                                                top_k=10,
                                                                candidate_count=1,
                                                                max_output_tokens=8000,
                                                                stop_sequences=["STOP!"]
                            ),
                            safety_settings=[SafetySetting(
                                                category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                                threshold=generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                                            ),
                                            SafetySetting(
                                                category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                                threshold=generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                                            ),
                                            SafetySetting(
                                                category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                                threshold=generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                                            ),
                                            SafetySetting(
                                                category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                                threshold=generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                                            ),
                            ],
                            tools=[query_tools]
)

if 'chat' not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(response_validation=True,
                                                            history=st.session_state.gemini_history)
def reset_conversation():
    del st.session_state.gemini_history
    del st.session_state.chat
    del st.session_state.messages

st.button(label='Reset', key='reset', on_click=reset_conversation)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar='🧑🏻' if message['role']=='user' else '🤖'):
        st.markdown(message["content"])  # noqa: W605
        try:
            with st.expander("Function calls, parameters, and responses"):
                st.markdown(message["backend_details"])
        except KeyError:
            pass

if prompt := st.chat_input(placeholder="Ask me about company policies, meeting minutes organization, or email writing assistance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='🧑🏻'):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar='🤖'):

        message_placeholder = st.empty()
        full_response = "" # pylint: disable=invalid-name
        response = st.session_state.chat.send_message(prompt)
        print(response)
        response = response.candidates[0].content.parts[0]

        backend_details = "" # pylint: disable=invalid-name
        api_requests_and_responses  = []

        function_calling_in_process = True # pylint: disable=invalid-name

        while function_calling_in_process:
            try:
                params = {}
                for key, value in response.function_call.args.items():
                    params[key] = value

                print(response.function_call.name)
                print(params)

                if response.function_call.name == "create_mom":
                    result, api_response = create_mom(**params)
                    st.write(result)
                    api_requests_and_responses.append([response.function_call.name, params, api_response])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "notes",
                            "notes": result,
                        }
                    )

                if response.function_call.name == "write_an_email":
                    email, api_response = write_an_email(**params) # pylint: disable=invalid-name
                    api_requests_and_responses.append([response.function_call.name, params, api_response])
                    st.write(email)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "email",
                            "email": email,
                        }
                    )

                if response.function_call.name == "answer_query":
                    answer, api_response = answer_query(**params)
                    api_requests_and_responses.append([response.function_call.name, params, api_response])

                print(api_response)

                response = st.session_state.chat.send_message(
                    Part.from_function_response(
                        name=response.function_call.name,
                        response={
                            "role": "assistant",
                            "content": api_response,
                        },
                    ),
                )

                backend_details += "- Function call:\n"
                backend_details += (
                    "   - Function name: ```"
                    + str(api_requests_and_responses[-1][0])
                    + "```"
                )
                backend_details += "\n\n"
                backend_details += (
                    "   - Function parameters: ```"
                    + str(api_requests_and_responses[-1][1])
                    + "```"
                )
                backend_details += "\n\n"
                backend_details += (
                    "   - Function API response: ```"
                    + str(api_requests_and_responses[-1][2])
                    + "```"
                )
                backend_details += "\n\n"

                with message_placeholder.container():
                    st.markdown(backend_details)

                response = response.candidates[0].content.parts[0]
                print(f"function return: {api_response}, model_response: {response}")
            except AttributeError:
                function_calling_in_process = False # pylint: disable=invalid-name

        st.session_state.gemini_history = st.session_state.chat.history
        # st.markdown(response.text)
        full_response = response.text

        with message_placeholder.container():
            st.markdown(full_response.replace("$", "\\$"))  # noqa: W605
            with st.expander("Function calls, parameters, and responses:"):
                st.markdown(backend_details)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )