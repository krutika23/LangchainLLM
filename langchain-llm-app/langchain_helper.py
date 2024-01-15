from langchain.llms import OpenAI, Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain
from langchain.agents import load_tools, AgentType, initialize_agent, Tool
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from getpass import getpass

COHERE_API_KEY = "1o82jVB1OkE05PIxZQSvPdn0beimCgs7j1q9rDHY" #getpass()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
model = AutoModelForCausalLM.from_pretrained("openai-gpt")

def generate_pet_name(animal_type,pet_color):
    llm= Cohere(cohere_api_key=COHERE_API_KEY) #OpenAI(temperature=0.6)
    # name=llm("Give me some cool dog names for a shetzu with brown and with fur")
    prompt = "Give me some cool names for a {animal_type} with {pet_color} color"
    # inputs = tokenizer(prompt, return_tensors="pt")
    # outputs = model.generate(**inputs, max_length=50, num_return_sequences=1, temperature=0.6,
    # do_sample=True)
    # name = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_template_name=PromptTemplate(
        input_variables=['animal_type','pet_color'],template=prompt,
    )
    print(prompt_template_name)
    name_chain=LLMChain(llm=llm,prompt=prompt_template_name,output_key="pet_name")
    response=name_chain({'animal_type':animal_type,'pet_color':pet_color})

    return response

def langchain_agent():
    llm=Cohere(cohere_api_key=COHERE_API_KEY,temperature=0.7)
    tools=load_tools(["wikipedia","llm-math"],llm=llm)

    agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True)

    result=agent.run("What's the average age of a dog? Multiply that age by 4")
    return result

def calculator(prompt):
    llm=Cohere(cohere_api_key=COHERE_API_KEY,temperature=0.7)
    llm_math=LLMMathChain(llm=llm)
    #initialize a math tool
    math_tool=Tool(name='Calculator',
    func=llm_math.run, description="Used to answer math questions") # Description helps the llm to understand when to use the tool, this description goes inside the prompt

    tools=[math_tool]
    zero_shot_agent=initialize_agent(tools=tools, agent="zero-shot-react-description",
    llm=llm,verbose=True,max_iterations=10)
    return zero_shot_agent(prompt)

def general_purpose_queries_and_math(prompt):
    llm=Cohere(cohere_api_key=COHERE_API_KEY,temperature=0.7)
    prompt_template=PromptTemplate(input_variables=['prompt'],
    template="{prompt}")
    llm_chain=LLMChain(llm=llm,prompt=prompt_template)
    math_tool=Tool(name='Calculator',func=LLMMathChain(llm=llm).run, description="Used to answer math questions")
    llm_tool=Tool(name="Language model",func=llm_chain.run,description="Use this tools for general purpose queries and logic")
    zero_shot_agent=initialize_agent(tools=[llm_tool,math_tool], agent="zero-shot-react-description",
    llm=llm,verbose=True,max_iterations=10)
    return zero_shot_agent(prompt)

#Making use of Tools and agents





if __name__=="__main__":
    # print(generate_pet_name(animal_type="dog",pet_color="brown"))
    # print(langchain_agent())
    # print(calculator("What is 37593 * 67?"))
    # print(calculator("If laura has 2 chocolates and James brings 4 and a half box of chocolates where every box has 4 chocolates, how many chocolates do they both have in total"))
    print(general_purpose_queries_and_math("What is the capital of norway? Also, tell me how much is 5*6?"))

