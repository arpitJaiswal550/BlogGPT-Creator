import os 
import streamlit as st 
# from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain_community.llms import Ollama

# response = llm.invoke("tell me a mathematics joke!", temprature=1)
# print(response)

# App framework
st.title('🦜🔗 Blog GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# prompt templates
title_tamplate = PromptTemplate(
    input_variables=['Topic'],
    template="write me a single blog title about {Topic}. Just give the blog title as the response."
)
blog_tamplate = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template="write me a blog based on this title TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}"
)

# Memory
title_memory = ConversationBufferMemory(input_key='Topic', memory_key='chat_history')
blog_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

#llms
llm = Ollama(model='llama3', temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_tamplate, verbose=True, output_key='title', memory=title_memory)
blog_chain = LLMChain(llm=llm, prompt=blog_tamplate, verbose=True, output_key='blog', memory=blog_memory)
# sequential_chain = SimpleSequentialChain(chains=[title_chain, blog_chain], verbose=True)

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    blog = blog_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(blog) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Blog History'): 
        st.info(blog_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)