from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import numpy as np
dict_results={}
new_dict_results={}
#prompt 1
template = """You're a smart AI assistant that should find all relevant keywords and contextual information that people might use to search for the historical objects described in the sentence provided.
               Use external knowledge to enrich the list of search terms with historical information, such as dates, places, battles, etc. List only these search terms, and number them. Do not use full sentences.
               sentence: {sentence}."""
#prompt 2
# template= """[INST] <<SYS>>
# You're a smart AI assistant that should find all relevant keywords and contextual information that people might use to search for the historical items described in the sentence provided.
#                 Use external knowledge to enrich the list of search terms with historical information such as events, locations, names of people, names of ships, military units, general topic.
#                 only add a keyword when it is provided in the sentence or when you can add it with external knowledge. When no keyword is added to a category, add 'UNKNOWN'
#                 Use the following formatting: 1."events": "event name (start date – end date)", 2."locations": "Place, County/State, Country (historic name)", 3."names of people": "Surname, Forenames Prefixes to surname (additions to name)", 4."names of ships": "HMS ship name (launch date-fate date)", 5."military units": "name (start date-end date)", 6. "general topic": topic of the sentence. 
#                 Answer by only listing this required information. Do not use full sentences and solely follow the formatting as shown above, separate multiple examples per category by a ','. 
# <</SYS>>
# sentence: {sentence}.[/INST]"""



# df_data=pd.read_csv(r'testset_dataset.csv') (uncomment if you have a full dataset you want to get keywords from)
list_temp=[1]
prompt = PromptTemplate(template=template, input_variables=["sentence"])
for i in list_temp:
    data=["painting about Nelson's death on the HMS Victory","napkin from 1914-1918"] #(comment this out if you want to use the full dataset)
    #df_data['temp'+str(i)]=np.nan #(uncomment if you want to use a full dataset)
    llm = CTransformers(model=r"llama-2-7b-chat.Q4_K_M.gguf", model_type='llama', config={"temperature":i})
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    for text in data: #(comment this out if you want to use the full dataset)
        keywords=llm_chain(text) #(comment this out if you want to use the full dataset)
        print(keywords) #(comment this out if you want to use the full dataset)

    #for row in df_data: (uncomment if you want to use a full dataset)
        # df_data['temp'+str(i)]= df_data.apply(lambda row: llm_chain(row['description'])['text'], axis=1) #(uncomment if you want to use a full dataset)
        # df_data.to_csv(r'filled_testset_dataset_prompt2.csv') #(uncomment if you want to use a full dataset)