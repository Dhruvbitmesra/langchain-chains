from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1=ChatOpenAI()

model2=ChatAnthropic()

prompt1=PromptTemplate(
    template="genrate a short simple summary about{topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="genrate 5 quiz question on {text}",
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='merge the provided notes and quiz into the single document->{notes} and {quiz} ',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser
}
)

merge_chain=prompt3|model1|parser

chain=parallel_chain|merge_chain

text=''' PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance. In scikit-learn, PCA is implemented as a transformer object that learns 
 components in its fit method, and can be used on new data to project it on these components.

PCA centers but does not scale the input data for each feature before applying the SVD. The optional parameter whiten=True makes it possible to project the data onto the singular space while scaling each component to unit variance. This is often useful if the models down-stream make strong assumptions on the isotropy of the signal: this is for example the case for Support Vector Machines with the RBF kernel and the K-Means clustering algorithm.
'''

result=chain.invoke({"text":text})

print(result)

chain.get_graph().print_ascii