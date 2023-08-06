#https://safe.menlosecurity.com/https://developers.generativeai.google/examples/doc_search_emb
import os
import google.generativeai as palm
import textwrap
import numpy as np
import pandas as pd

palm.configure(api_key=os.environ["PALM_API_KEY"])
embedding_model = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods][0]
text_model = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods][0]

# Get the embeddings of each text and add to an embeddings column in the dataframe
def embed_fn(text):
    return palm.generate_embeddings(model=embedding_model, text=text)['embedding']

def find_best_passage(query, dataframe):
    """
    Compute the distances between the query and each document in the dataframe
    using the dot product.
    """
    query_embedding = palm.generate_embeddings(model=embedding_model, text=query)
    #print(len(query_embedding['embedding']))
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding['embedding'])

    #idx = np.argmax(dot_products)
    idx_topn = np.argsort(dot_products)[::-1][:4]  # provide top 4 docs found
    tmp_text = ""
    
    for idx in idx_topn:
        tmp_text = tmp_text + str(dataframe.iloc[idx]['context']) + "\n\n"
    
    return tmp_text # Return text from index with max value


def make_prompt(query,df):
    relevant_passage=find_best_passage(query,df)
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent("""You are the People's Association's Finance Division's assistant. Your role is to provide comprehensive and accurate information to officers regarding financial matters using text from the reference passages included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    If the passage is irrelevant to the answer, say "answer not available in context".
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)
    print(prompt)
    return prompt

def answer_query_with_context(query,df):
    answer = palm.generate_text(prompt=make_prompt(query,df),
                                model=text_model,
                                candidate_count=1,
                                temperature=0,
                                max_output_tokens=500)
    print(answer.result)
    return answer.result
    # return answer.result.strip(" \n").replace('\n', '<br />')
