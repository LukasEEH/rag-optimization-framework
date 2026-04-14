import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import os
import time
import json
import datetime
import asyncio
import numpy as np
from os.path import join
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import SingleTurnMetric
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM

import metrics_provider

import sys
sys.path.insert(1, '../rag/baseline')
from rag import RAG


CONFIG =  {
    "OUTPUT_DIR": ".",
    "TOP_K": 4,
    "EMBED_MODEL_ID": "qwen3-embedding:8b",
    "LLM_ID": "gpt-oss:20b",
    "MAX_RETRIES": 3,
    "CONFIG_PATH": "config.dev.json",
    "QNA_INFOS_PATH": "data/qna_infos.json"
}


logger.info("🧪 INITIALIZE SYSTEM UNDER TEST...")
rag = RAG(CONFIG['CONFIG_PATH'])
logger.info("✅ SUCCESSFULLY INITIALIZED SYSTEM UNDER TEST.")


def get_passed_seconds_since(start_time: float) -> float:
    elapsed_seconds = time.time() - start_time
    return elapsed_seconds


def get_time_log(start_time: float) -> str:
    elapsed_time = get_passed_seconds_since(start_time)
    return f"{round(elapsed_time, 2)} sec (~ {round(elapsed_time/60, 2)} min)"


def save_results(results: list[dict], output_dir: str):
    logger.info(f"💾 Saving results to {output_dir}...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json"
        filepath = join(output_dir, filename)

        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)

        logger.info(f"Saved benchmarking results to {filepath} successfully.")
    except (IOError, PermissionError) as e:
        logger.error(f"Failed to save results to {filepath}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during file saving: {e}")


async def acreate_sample(query: str, ground_truth: str) -> tuple[SingleTurnSample, float]:
    creation_start = time.time()

    logger.info("Generate response with RAG module...")
    response, context_nodes = await rag.respond_verbose(query)

    context = [entry.text for entry in context_nodes]

    generation_time = get_passed_seconds_since(creation_start)

    sample = SingleTurnSample(
        user_input=str(query),
        retrieved_contexts=context,
        response=str(response.message.content),
        reference=str(ground_truth)
    )

    logger.info(f"Finished creating sample in {generation_time}s")

    return sample, generation_time


async def acreate_all_samples(qnas: list[dict]) -> list[tuple[SingleTurnSample, float]]:
    samples_tasks = []
    logger.info(f"💬 Starting sample creation for {len(qnas)} question-answer pairs in parallel...")

    for qna_pair in qnas:
        samples_tasks.append(
            acreate_sample(qna_pair['question'], qna_pair['answer'])
        )

    # unpacking the task elements for gathering
    results = await asyncio.gather(*samples_tasks)
    logger.info(f"✅ Succesfully created {len(qnas)} sample(s).")

    return results


async def acalculate_metric(metric: SingleTurnMetric, sample: SingleTurnSample) -> float:
    last_exception = None
    for attempt in range(CONFIG["MAX_RETRIES"]):
        calculation_start = time.time()
        try:
            logger.info(f"Calculating {metric.name} (attempt {attempt+1})")
            score = await metric.single_turn_ascore(sample)
            
            if np.isnan(score):
                logger.warning(f"Recieved nan for {metric.name} at attempt {attempt+1}.")
                continue
            
            logger.info(f"Successfully calculated {metric.name} in {get_time_log(calculation_start)}.")
            return float(score)
        except Exception as e:
            last_exception = e
            logger.warning(f"Failed calculating {metric.name} in {get_time_log(calculation_start)} at attempt {attempt+1}: {e}")
            await asyncio.sleep(3 * (2 ** attempt))
    logger.error(f"Failed calculating {metric.name} after {CONFIG["MAX_RETRIES"]} attempts, last exception: {last_exception}")
    return float('nan')


async def acalculate_all_metrics_for_sample(
    sample_tuple: tuple[SingleTurnSample, float], metrics: list[SingleTurnMetric]
) -> dict[str: any]:
    calculation_start = time.time()

    sample, generation_time = sample_tuple

    result = {
        "generation_time": generation_time,
        "user_input": sample.user_input,
        "retrieved_contexts": sample.retrieved_contexts,
        "response": sample.response,
        "reference": sample.reference,
    }

    metrics_tasks = []
    metrics_names = []

    for metric in metrics:
        metrics_tasks.append(acalculate_metric(metric, sample))
        metrics_names.append(metric.name)

    scores =  await asyncio.gather(*metrics_tasks)

    for i, name in enumerate(metrics_names):
        result[name] = scores[i]

    logger.info(f"Finished scoring all metrics for sample in {get_time_log(calculation_start)}.")
    return result


async def acalculate_all_metrics_for_samples(
    samples: list[tuple[SingleTurnMetric, float]], metrics: list[SingleTurnMetric]
) -> list[dict[str, any]]:
    logger.info(f"📏 Starting to score {len(samples)} samples on {len(metrics)} metrics each...")

    tasks = [
        acalculate_all_metrics_for_sample(sample, metrics) for sample in samples
    ]

    results = await asyncio.gather(*tasks)
    return results


async def main():
    pipeline_start = time.time()
    logger.info(f"🚀 STARTING BENCHMARK: setting things up with config {json.dumps(CONFIG)}")

    ollama_llm: BaseRagasLLM = LangchainLLMWrapper(
        ChatOllama(model=CONFIG["LLM_ID"], temperature=0, max_tokens=8192)
    )
    ollama_embeddings: BaseRagasEmbeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=CONFIG["EMBED_MODEL_ID"]))

    logger.info(f"Loading Q&A pairs from {CONFIG["QNA_INFOS_PATH"]}...")
    with open(CONFIG["QNA_INFOS_PATH"]) as f:
        qna_infos = json.load(f)

    logger.info(f"Successfully loaded the Q&A pairs for {len(qna_infos)} datasets.")

    logger.info("Loading the metrics...")
    metrics = metrics_provider.get_metrics(ollama_llm, ollama_embeddings)
    logger.info(f"Successfully loaded {len(metrics)} metrics: {[m.name for m in metrics]}")

    results = []

    for dataset in qna_infos['datasets']:
        logger.info(f"🔬 Evaluate {dataset['name']} tasks...")
        dataset_results = []
        samples = await acreate_all_samples(dataset['qnas'])
        dataset_results = await acalculate_all_metrics_for_samples(samples, metrics)
        results.append({ "dataset": dataset['name'], "results": dataset_results })
        logger.info(f"✅ Finished scoring {dataset['name']} tasks.")

    logger.info(f"Finalize results...")

    benchmark_infos = {
        "language_model": CONFIG['LLM_ID'],
        "embedding_model": CONFIG['EMBED_MODEL_ID'],
        "duration": get_time_log(pipeline_start),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }

    save_results(benchmark_infos, CONFIG["OUTPUT_DIR"])
    logger.info(f"🏁 FINISHED BENCHMARK in {get_time_log(pipeline_start)}!")


asyncio.run(main())