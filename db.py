import os
import clickhouse_connect

import pandas as pd
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import configparser


config = configparser.ConfigParser()
config.read('config.INI')

# Accessing configuration values
api_key = config.get('General', 'GEMINI_API_KEY')
username = config.get('General', 'DB_USERNAME')
password = config.get('General', 'DB_PASSWORD')

genai.configure(api_key=api_key)

class VectorDB:
    def __init__(self):
        self.loader = PyPDFLoader("req/Arjtech Private Ltd. Company Policy Document.pdf")

        self.dbclient = clickhouse_connect.get_client(
            host='myscaledb',
            port='port',
            username= username,
            password= password
            )

        self.pages = None

    @staticmethod
    def get_embeddings(text):

        model = 'models/embedding-001'
        embedding = genai.embed_content(model=model,
                                        content=text,
                                        task_type="retrieval_document")

        return embedding['embedding']

    def read_and_chunkize_text(self):

        pages = self.loader.load_and_split()
        text = "\n".join([doc.page_content for doc in pages])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents([text])
        for i, d in enumerate(docs):
            d.metadata = {"doc_id": i}

        return docs

    def create_embedding_df(self):

        docs = self.read_and_chunkize_text()

        content_list = [doc.page_content for doc in docs]
        embeddings = [self.get_embeddings(content) for content in content_list]

        dataframe = pd.DataFrame({'page_content': content_list
        , 'embeddings': embeddings})

        return dataframe

    def create_and_update_vectordb(self):

        try:

            self.dbclient.command("""CREATE TABLE default.handbook (
            id Int64, page_content String, embeddings Array(Float32),
            CONSTRAINT check_data_length CHECK length(embeddings) = 768) 
            ENGINE = MergeTree() ORDER BY id""")

            dataframe = self.create_embedding_df()

            batch_size = 10
            num_batches = len(dataframe) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_data = dataframe[start_idx:end_idx]
                # Insert the data
                self.dbclient.insert("default.handbook", batch_data.to_records(index=False).tolist(), column_names=batch_data.columns.tolist())
                print(f"Batch {i+1}/{num_batches} inserted.")

            # Create a vector index for a quick retrieval of data
            self.dbclient.command("""
            ALTER TABLE default.handbook
                ADD VECTOR INDEX vector_index embeddings
                TYPE MSTG
            """)

        except Exception as e:
            return f"Error in Creating VectorDB :: {e}"

        return "VECTORDB CREATED SUCCESFULLY"

    def get_relevant_docs(self, user_query):

        query_embeddings = self.get_embeddings(user_query)

        results = self.dbclient.query(f"""
            SELECT page_content,
            distance(embeddings, {query_embeddings}) as dist FROM default.handbook ORDER BY dist LIMIT 3
        """)

        relevant_docs = []
        for row in results.named_results():
            relevant_docs.append(row['page_content'])

        return relevant_docs

# vb = VectorDB()
# response = vb.create_and_update_vectordb()
# print(response)
# df = vb.create_embedding_df()
# print("embedding df\n", df.head())

# print(vb.get_relevant_docs("Dress Code?"))
