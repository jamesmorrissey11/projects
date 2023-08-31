from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from tools.parsing import CustomOutputParser
from tools.prompts import CustomPromptTemplate
from tools.template import search_template_with_history


def create_multi_search_agent(tools, tool_names):
    """
    Create agent executor with access to web and vectorstore search tools
    """
    prompt_with_history = CustomPromptTemplate(
        template=search_template_with_history,
        tools=tools,
        input_variables=["input", "intermediate_steps", "history"],
    )
    llm_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        prompt=prompt_with_history,
    )
    output_parser = CustomOutputParser()
    tool_agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )
    tool_memory = ConversationBufferWindowMemory(k=2)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=tool_agent, tools=tools, verbose=True, memory=tool_memory
    )
    return agent_executor
