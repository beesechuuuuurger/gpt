import langchain
import pandas as pd
from serpapi import GoogleSearch
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.indexes import VectorstoreIndexCreator
import os

os.environ["OPENAI_API_KEY"] = "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"

loader = CSVLoader(file_path='D:\\DEV\\EvolveGPT\\input.csv')

index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key = "Question", output_key = "Answer")

query = {"Question": "What would be the best AWS cloud native Solutions for these apps?"}
response = chain(query)
print(response)

def search_google(response): 
    if response == "I don't know." :
        params = {
            "api_key": "d2d6f2c4bd87f8d2959bb0b0e0f0e81d5387521acb0ff0b1d211b2546c092e29",
            "engine": "google",
            "q": response
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        print(results)
        return results
        
    else:
        return response
