from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Cevap soruyu karşılıyor mu, 'evet' veya 'hayır'"
    )


llm = ChatOpenAI(temperature=0)

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """Siz, verilen bir cevabın soruyu karşılayıp karşılamadığını veya çözüme kavuşturup kavuşturmadığını değerlendiren bir hakemsiniz. \n
'evet' veya 'hayır' şeklinde ikili bir puan verin. 'Evet', cevabın soruyu çözdüğü/yanıtladığı anlamına gelir."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Soru: {question} \n\n LLM üretimi: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
