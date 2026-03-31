from dataclasses import dataclass, field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
import typing as t
from ragas.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field

class CorrectnessInput(BaseModel):
    user_input: str = Field()
    response: str = Field()
    reference: str = Field()

class CorrectnessOutput(BaseModel):
    correctness: str = Field()

class CorrectnessPrompt(PydanticPrompt[CorrectnessInput, CorrectnessOutput]):
    instruction = """You are an expert evaluator specializing in Factual Correctness, Knowledge Verification, and Information Accuracy. Your task is to act as a judge for a Retrieval-Augmented Generation (RAG) system.

    Evaluate and score the response regarding its Correctness.
    
    ### Input Data

    **User Question:**
    {user_input}

    **Ground-Truth Answer (Reference):** 
    {reference}

    **Generated Answer (Response):** 
    {response}

    ### Evaluation Criteria & Scoring Rubric

    To determine your score, you must assess the following four key aspects. Think step-by-step through each one before making your final judgment.

    1.  Semantic Accuracy: 
        Does the generated response convey the same factual information as the ground-truth reference? Are the key facts, dates, names, and technical details identical in meaning?

    2.  Alignment:
        Do the claims of the generated answer align with the claims of the reference? Do they factually overlap?

    4.  Absence of Contradictions: 
        Does any part of the generated response directly contradict the ground-truth reference? Even if most of the answer is correct, a single factual contradiction significantly lowers correctness.
    
    3.  Relevance and Focus: 
        Does the response stay focused on the user's question as defined by the reference, or does it include "hallucinated" extra information that, while perhaps true in the real world, is not supported by the provided reference?


    ### Scoring Guide

    Use this holistic rubric to assign your final score:

    - **Score 0-2 (Totally Incorrect):** The response is completely wrong, irrelevant, or contradicts the ground truth in its entirety.

    - **Score 3-4 (Major Errors):** The response contains significant factual errors or misses more than 50% of the key points mentioned in the reference. It may contain a mix of correct and fundamentally incorrect statements.

    - **Score 5-6 (Partially Correct):** The response captures the "gist" or the main idea of the reference but misses several details, nuances, or secondary points. It is technically correct in what it states but is incomplete.

    - **Score 7-8 (Mostly Correct):** The response is accurate and covers all key points from the reference. There may be very minor omissions, unnecessary additional information or slight differences in phrasing that do not affect the factual integrity of the answer.

    - **Score 9-10 (Perfectly Correct):** The response is entirely accurate, complete, and aligns perfectly with the ground-truth reference. It answers the user’s question fully and concisely without any contradictions or unnecessary filler.

    ### Required Output

    After your thorough evaluation, provide **only the final integer score**. Do not include any explanations, reasoning, or additional text. Your entire response must be a single number.

    Final Score:"""
    input_model = CorrectnessInput
    output_model = CorrectnessOutput

@dataclass
class Correctness(MetricWithLLM, SingleTurnMetric):
    # name of the metric
    name: str = "correctness"

    correctness_prompt = CorrectnessPrompt()

    # required columns for the metric
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        prompt_input = CorrectnessInput(
            user_input = sample.user_input, response = sample.response, reference = sample.reference
        )
        try:
            first_response = await self.correctness_prompt.generate(
                data=prompt_input, llm=self.llm
            )
            second_response = await self.correctness_prompt.generate(
                data=prompt_input, llm=self.llm
            )
            first_score = int(first_response.correctness)
            second_score = int(second_response.correctness)
            mean_score = float(first_score + second_score) / 2
            return mean_score/10.0
        except Exception as e:
            print(f"error while calculating correctness score: {e}")
            return float('nan')