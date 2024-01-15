import os
from langchain.llms import Cohere
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine, insert
from sqlalchemy import Column, Integer, String, Table, Date, Float
from datetime import datetime



load_dotenv()

COHERE_API_KEY=os.getenv("COHERE_API_KEY")
llm=Cohere(cohere_api_key=COHERE_API_KEY,temperature=0.7)

#Agents help LLM use some external tools , the agents decide the order of actions and which tool to use when

#let's create a database tool which the agent can use 
metadata_=MetaData()

#create stocks table
stocks = Table("stocks",metadata_,
    Column("obs_id", Integer, primary_key=True),
    Column("ticker", String(4), nullable=False),
    Column("price", Float, nullable=False),
    Column("date", Date, nullable=False),
)

engine = create_engine("sqlite:///:memory:")
metadata_.create_all(engine)

observations = [
    [1, 'ABC', 200, datetime(2023, 1, 1)],
    [2, 'ABC', 208, datetime(2023, 1, 2)],
    [3, 'ABC', 232, datetime(2023, 1, 3)],
    [4, 'ABC', 225, datetime(2023, 1, 4)],
    [5, 'ABC', 226, datetime(2023, 1, 5)],
    [6, 'XYZ', 810, datetime(2023, 1, 1)],
    [7, 'XYZ', 803, datetime(2023, 1, 2)],
    [8, 'XYZ', 798, datetime(2023, 1, 3)],
    [9, 'XYZ', 795, datetime(2023, 1, 4)],
    [10, 'XYZ', 791, datetime(2023, 1, 5)],
]

def insert_obs(table):
    for obs in table:
        stmt = insert(stocks).values(obs_id=obs[0],ticker=obs[1],price=obs[2],date=obs[3])
        with engine.begin() as conn:
            conn.execute(stmt)

insert_obs(observations)

db = SQLDatabase(engine)

#Create an SQL DB utility chain and pass llm and database 
sql_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

#Now we have our llm, tool and we will define the type of agent we want to use , an SQL agent using SQLToolkit
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=3 # this prevents the agent from executing infinite processing steps 
)

#Zero_shot agents dont have a memory though, for that we can use conversational react agents 

def count_tokens(agent, query):
    result = agent(query)
    return result

result = count_tokens(
    agent_executor,
    "What is the multiplication of the ratio between stock " +
    "prices for 'ABC' and 'XYZ' in January 3rd and the ratio " +
    "between the same stock prices in January the 4th?"
)

#The agent takes steps in the format 1.Thought 2.Action 3.Action inputs and 4.Observation 
#The LLM now has the ability to reason on how to best use tools 
print(result)
print("------------------------------------------------------------------------------------------------------------")
print(agent_executor.agent.llm_chain.prompt.template)






