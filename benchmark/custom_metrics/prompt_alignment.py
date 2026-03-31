from dataclasses import dataclass, field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
import typing as t
from ragas.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field

class PromptAlignmentInput(BaseModel):
    user_input: str = Field()
    response: str = Field()

class PromptAlignmentOutput(BaseModel):
    prompt_alignment: str = Field()

class PromptAlignmentPrompt(PydanticPrompt[PromptAlignmentInput, PromptAlignmentOutput]):
    instruction = """You are an expert evaluator specializing in the **Prompt Alignment** of AI systems. Your task is to act as a judge for a Retrieval-Augmented Generation (RAG) system.

    Evaluate and score the response regarding its prompt alignment.

    ### Input Data

    **User Question:**
    {user_input}

    **Generated Answer:**
    {response}

    ### Evaluation Criteria & Scoring Rubric

    To determine your score, you must assess the following five key aspects. Think step-by-step through each one before making your final judgment.

    1.  Precision:
        Does the answer directly address the core question asked or the core issue addressed, or does it drift into tangentially related topics?

    2.  Completeness:
        Does the model address all of the issues addressed from the prompt?

    3.  Output Format:
        Does the output match the requested structure? (e.g., JSON, Markdown table, single paragraph, bullet points)

    4.  Avoidance:
        Does the model avoid things it was told not to do?

    5.  Language Consistency:
        Does the model use consistent language to the user Input?

    ### Scoring Guide

    Use this holistic rubric to assign your final score:

    - **Score 0-2 (Very Poor)**: The response is irrelevant or a complete refusal. The system addresses a completely different topic than requested, offers a generic refusal without valid safety grounds, or hallucinates constraints that do not exist. It is entirely unusable for the user's intent.
    - **Score 3-4 (Poor)**: The response represents a major alignment failure. It attempts to address the topic but misses the core objective (e.g. summarizing a text when asked to translate it). Critical constraints are ignored. The user would need to edit the response significantly or reprompt to get what they asked for.
    - **Score 5-6 (Average)**: The response is partially aligned. The core question is answered, but the system ignores secondary instructions (e.g., word count, formatting specificities, or negative constraints like "do not use lists").
    - **Score 7-8 (Good)**: The response is highly compliant and relevant. All major instructions and the core intent are addressed correctly. There may be minor slips in nuance, tone, or very specific formatting details (e.g., using bullet points instead of numbered lists), but the output is immediately useful with minimal tweaking.
    - **Score 9-10 (Excellent/Perfect)**: The response is flawlessly aligned. The system captures the explicit instructions and implicit intent perfectly. It adheres to all complex constraints, including format, length, persona, and negative constraints. A score of 10 represents a flawless example of prompt alignment.

    ### Required Output

    After your thorough evaluation, provide **only the final integer score**. Do not include any explanations, reasoning, or additional text. Your entire response must be a single number.

    Final Score:"""
    input_model = PromptAlignmentInput
    output_model = PromptAlignmentOutput

@dataclass
class PromptAlignment(MetricWithLLM, SingleTurnMetric):
    # name of the metric
    name: str = "prompt_alignment"

    prompt_alignment_prompt = PromptAlignmentPrompt()

    # required columns for the metric
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response"}
        }
    )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        prompt_input = PromptAlignmentInput(
            user_input = sample.user_input, response = sample.response, retrieved_contexts = sample.retrieved_contexts
        )
        try:
            first_response = await self.prompt_alignment_prompt.generate(
                data=prompt_input, llm=self.llm
            )
            second_response = await self.prompt_alignment_prompt.generate(
                data=prompt_input, llm=self.llm
            )
            first_score = int(first_response.prompt_alignment)
            second_score = int(second_response.prompt_alignment)
            mean_score = float(first_score + second_score) / 2
            return mean_score/10.0
        except Exception as e:
            print(f"error while calculating prompt_alignment score: {e}")
            return float('nan')