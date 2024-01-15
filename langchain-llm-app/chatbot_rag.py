from json import load
from langchain.chat_models import ChatCohere
from langchain.embeddings import CohereEmbeddings

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.vectorstores import Pinecone
import pinecone
import time
from tqdm.auto import tqdm

#Used to load Cohere api key from os.environ
load_dotenv()

chat = ChatCohere()

messages = [HumanMessage(content="knock knock")]

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to understand string theory in two lines.")
]

# res=chat(messages)
# # print("First response:\n ",res," \n\n")

# messages.append(AIMessage(content=res.content) )

# #Create a ne user prompt for a followup question
# prompt = HumanMessage(
#     content="Why do physicists believe it can produce a 'unified theory'?"
# )
# messages.append(prompt)
# print(messages)

#Next AI ressponse to that
# res= chat(messages)

# print("Second response:\n",res.content)

#RAG implementation using VectorDB


#Dataset of LLama2 archive papers 
dataset=load_dataset("jamescalam/llama-2-arxiv-papers-chunked",split="train")

pinecone.init(api_key="97b9cb52-f2de-4d0b-baf4-e8987aa0b849",environment="gcp-starter")

#Create index 

index_name="llama-3-rag"
print("does the index exist? ",index_name in pinecone.list_indexes())

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name,dimension=384,metric="cosine")

    #wait for the index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pinecone.Index(index_name)
print("index status: ",index.describe_index_stats())

embed_model = CohereEmbeddings(model="embed-english-light-v3.0") #OpenAIEmbeddings(model="text-embedding-ada-002")

data = dataset.to_pandas() 

batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]

    # generate unique ids for each chunk
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    # get text to embed
    texts = [x['chunk'] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['chunk'],
         'source': x['source'],
         'title': x['title']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

text_field = "text"  # the metadata field that contains our text

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

query = "What is so special about Llama 2?"

# docs=vectorstore.similarity_search(query, k=3)

def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

# print(augment_prompt(query))

# create a new user prompt
prompt = HumanMessage(
    content=augment_prompt(query)
)
# add to messages
messages.append(prompt)

res = chat(messages)

print("First response: ",res.content)
prompt = HumanMessage(
    content="what safety measures were used in the development of llama 2?"
)

res = chat(messages + [prompt])
print("Second response: ",res.content)

prompt = HumanMessage(
    content=augment_prompt(
        "what safety measures were used in the development of llama 2?"
    )
)

res = chat(messages + [prompt])
print("Third response: ",res.content)

