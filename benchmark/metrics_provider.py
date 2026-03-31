from ragas.metrics.base import SingleTurnMetric
from custom_metrics.trustworthiness import Trustworthiness
from custom_metrics.prompt_alignment import PromptAlignment
from custom_metrics.correctness import Correctness
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    ResponseGroundedness,
    FactualCorrectness,
    AnswerRelevancy,
    SemanticSimilarity)
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings

def get_metrics(llm: BaseRagasLLM, embeddings: BaseRagasEmbeddings) -> list[SingleTurnMetric]:
    return [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        ContextRelevance(llm=llm),
        AnswerRelevancy(embeddings=embeddings, llm=llm),
        ResponseGroundedness(llm=llm),
        SemanticSimilarity(embeddings=embeddings),
        # FactualCorrectness(llm=llm),
        Correctness(llm=llm),
        Trustworthiness(llm=llm),
        PromptAlignment(llm=llm)
    ]