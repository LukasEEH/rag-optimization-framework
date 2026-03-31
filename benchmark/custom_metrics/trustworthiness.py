from dataclasses import dataclass, field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
import typing as t
from ragas.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field

class TrustworthinessInput(BaseModel):
    user_input: str = Field()
    response: str = Field()
    retrieved_contexts: list[str] = Field()

class TrustworthinessOutput(BaseModel):
    trustworthiness: str = Field()

class TrustworthinessPrompt(PydanticPrompt[TrustworthinessInput, TrustworthinessOutput]):
    instruction = """You are an expert evaluator specializing in the **Explainability, Transparency, and Traceability** of AI systems. Your task is to act as a judge for a Retrieval-Augmented Generation (RAG) system.

    Evaluate and score the response regarding its Trustworthiness.

    ### Input Data

    **User Question:**
    {user_input}

    **Retrieved Contexts:**
    {retrieved_contexts}

    **Generated Answer:**
    {response}

    ### Evaluation Criteria & Scoring Rubric

    To determine your score, you must assess the following five key aspects. Think step-by-step through each one before making your final judgment.

    1.  Citation and Grounding:
        Is every claim made in the Generated Answer directly supported by information present in the Retrieved Contexts?
        Does the answer include explicit and accurate citations (e.g., `[1]`, `[source A]`) that link statements to their specific source?

    2.  Traceability of Reasoning:
        Can a user logically follow how the system synthesized information from the retrieved contexts to formulate the generated answer? Is the connection between source and summary clear?

    3.  Support for User Fact-Checking:
        Does the system provide clear references, links, or document names that allow a user to easily find and verify the source of the information?

    4.  Chain-of-Reasoning Clarity:
        Considering the entire process (`query` -> `retrieval` -> `generation`), how transparent is the system's operation through it's response? Is it a "black box," or is the process understandable?

    5.  Entity Consistency:
        Compare the names of people, places, organizations, etc., in the generated answer with those in the retrieved contexts. Are they consistent and used correctly, ensuring user comprehension?

    ### Scoring Guide

    Use this holistic rubric to assign your final score:

    - **Score 0-2 (Very Poor)**: The answer is dominated by ungrounded, hallucinated, or factually incorrect information. Citations are completely missing or wildly inaccurate, making the answer highly unreliable.
    - **Score 3-4 (Poor)**: Significant portions of the answer (20−40%) contain ungrounded or clearly contradictory claims/data points (Hallucination). Citations are largely absent or frequently misattributed, making verification difficult.
    - **Score 5-6 (Average)**: Most claims (>80%) are grounded. Minor instances of hallucination (ungrounded or factually incorrect claims) are present, but they do not alter the overall message. Citations are sporadic or occasionally mislinked but cover the main points.
    - **Score 7-8 (Good)**: The vast majority (> 90%) of claims in the answer are well-grounded. Any ungrounded claims are minor and non-critical. Claims are supported by explicit citations. The reasoning is mostly traceable, and the user can fact-check the information with ease.
    - **Score 9-10 (Excellent/Perfect)**: Every claim and specific data point in the Generated Answer is directly and explicitly supported by one or more Retrieved Contexts. Citations are present for every supporting statement and are perfectly accurate (linking to the correct source segment), making fact-checking trivial. No claims are ungrounded or contradictory. A score of 10 represents a flawless example of explainability.

    ### Required Output

    After your thorough evaluation, provide **only the final integer score**. Do not include any explanations, reasoning, or additional text. Your entire response must be a single number.

    Final Score:"""
    input_model = TrustworthinessInput
    output_model = TrustworthinessOutput

@dataclass
class Trustworthiness(MetricWithLLM, SingleTurnMetric):
    # name of the metric
    name: str = "trustworthiness"

    trustworthiness_prompt = TrustworthinessPrompt()

    # required columns for the metric
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        prompt_input = TrustworthinessInput(
            user_input = sample.user_input, response = sample.response, retrieved_contexts = sample.retrieved_contexts
        )
        try:
            first_response = await self.trustworthiness_prompt.generate(
                data=prompt_input, llm=self.llm
            )
            second_response = await self.trustworthiness_prompt.generate(
                data=prompt_input, llm=self.llm
            )
            first_score = int(first_response.trustworthiness)
            second_score = int(second_response.trustworthiness)
            mean_score = float(first_score + second_score) / 2
            return mean_score/10.0
        except Exception as e:
            print(f"error while calculating trustworthiness score: {e}")
            return float('nan')