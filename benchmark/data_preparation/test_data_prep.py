import pandas as pd
import json
import math
import docx
import fpdf
import os

questions_file = "qna_infos.json"
base_dir = "../data"
docs_dir = f"{base_dir}/docs"

class Record():
    question: str
    answer: str
    context: str

    def __init__(self, question: str, answer: str, context: str):
        self.question = question
        self.answer = answer
        self.context = context


class Dataset():
    name: str
    records: list[Record]

    def __init__(self, name: str, records: list[Record]):
        self.name = name
        self.records = records


### load general QA dataset
qa_df = pd.read_json("hf://datasets/DiscoResearch/germanrag/germanrag.jsonl", lines=True)

result_df = qa_df.query('positive_ctx_idx == 0').head(1)
result_df = pd.concat([result_df, qa_df.query('positive_ctx_idx == 1').head(2)])
result_df = pd.concat([result_df, qa_df.query('positive_ctx_idx == 2').head(2)])
result_df = pd.concat([result_df, qa_df.query('positive_ctx_idx == 3').head(1)])
result_df = pd.concat([result_df, qa_df.query('positive_ctx_idx == -1').head(2)])

records: list[Record] = []

for i, row in result_df.iterrows():
    records.append(
        Record(question=row['question'], answer=row['answer'], context='. '.join(row['contexts']))
    )

qa_ds = Dataset("general QA", records)


### load multiple-choice dataset
mc_df = pd.read_json("hf://datasets/facebook/belebele/data/deu_Latn.jsonl", lines=True)

def get_multiple_choice_prompt(question, context, answer1, answer2, answer3, answer4):
    return f'''Beantworte die folgende Multiple-Choice Frage basierend auf dem abgerufenen Kontext mit einer der gegebenen Antworten.
    Frage: {question}
    Antwort 1: {answer1};
    Antwort 2: {answer2};
    Antwort 3: {answer3};
    Antwort 4: {answer4};'''

def get_multiple_choice_answer(correct_answer_num, answer):
    return f'Antwort {correct_answer_num}: {answer}'

result_df = mc_df.query('correct_answer_num == 1').head(1)
result_df = pd.concat([result_df, mc_df.query('correct_answer_num == 2').head(1)])
result_df = pd.concat([result_df, mc_df.query('correct_answer_num == 3').head(1)])
result_df = pd.concat([result_df, mc_df.query('correct_answer_num == 4').head(1)])

records: list[Record] = []

for i, row in result_df.iterrows():
    records.append(
        Record(
            question=get_multiple_choice_prompt(row['question'], row['flores_passage'], row['mc_answer1'], row['mc_answer2'], row['mc_answer3'], row['mc_answer4']), 
            answer=get_multiple_choice_answer(row['correct_answer_num'], row[f'mc_answer{row["correct_answer_num"]}']), 
            context=row['flores_passage']
        )
    )

mc_ds = Dataset("multiple choice", records)


### load fact-checking dataset
fc_df = pd.read_parquet("hf://datasets/tdiggelm/climate_fever/data/test-00000-of-00001.parquet")

def get_fact_check_prompt(claim):
    return f'Fact check the following claim: {claim}. '\
        'Answer with "0" if the claim is supported by the retrieved context, '\
        '"1" if the claim is not supported and '\
        '"2" if there is not enough provided information in the context to verify the claim properly'\
        'Provide only the according number, do not include any explanation, reasoning or additional text.'

result_df = fc_df.query('claim_label == 0').head(2)
result_df = pd.concat([result_df, fc_df.query('claim_label == 1').head(2)])
result_df = pd.concat([result_df, fc_df.query('claim_label == 2').head(2)])

records: list[Record] = []

for i, row in result_df.iterrows():
    context = ''
    for ev in row['evidences']:
        context += '. ' + ev['evidence']
    records.append(
        Record(
            question=get_fact_check_prompt(row['claim']),
            answer=row['claim_label'],
            context=context
        )
    )

fc_ds = Dataset("fact checking", records)
    

### load multi-hop QA dataset
mh_df = pd.read_json("hf://datasets/dgslibisey/MuSiQue/musique_ans_v1.0_dev.jsonl", lines=True)


target_ids = [
    '2hop__21075_5028',
    '2hop__736167_74735',
    '2hop__61714_89309',
    '3hop1__139787_88110_77129',
    '3hop1__691197_15840_36014',
    '4hop1__88342_75218_128008_86588',
    '4hop1__178366_229349_66759_75165'
]
result_df = mh_df.query('id in @target_ids')

records: list[Record] = []

for i, row in result_df.iterrows():
    context = ''
    for par in row['paragraphs']:
        context += '. ' + par['title'] + '. ' + par['paragraph_text']
    records.append(
        Record(question=row['question'], answer=row['answer'], context=context)
    )

mh_ds = Dataset("multi-hop QA", records)


### create qna_info JSON structure
datasets : list[Dataset] = [
    qa_ds, mc_ds, fc_ds, mh_ds
]

full_context = ""
qna_infos = {
    "datasets": []
}

for dataset in datasets:
    qnas = []
    for record in dataset.records:
        full_context += '\n' + record.context
        qnas.append(
            { "question": record.question, "answer": record.answer }
        )
    qna_infos['datasets'].append(
        { "name": dataset.name, "qnas": qnas }
    )

### add corporate QA questions
sse_qa = {
    "name": "corporate QA",
    "qnas": [
        # corporate QA paires were removed
    ]
}

# qna_infos['datasets'].append(sse_qa)


### split data and distribute to different files 
full_length = len(full_context)
part_length = math.floor(full_length / 3)

part1_end = part_length
part2_end = part1_end + part_length

part1 = full_context[:part1_end]
part2 = full_context[part1_end:part2_end]
part3 = full_context[part2_end:]

print(f"Part 1 (PDF) Length: {len(part1)}")
print(f"Part 2 (DOCX) Length: {len(part2)}")
print(f"Part 3 (TXT) Length: {len(part3)}")

pdf_path = f"{docs_dir}/bm_context_1.pdf"
docx_path = f"{docs_dir}/bm_context_2.docx"
txt_path = f"{docs_dir}/bm_context_3.txt"

if not os.path.exists(base_dir):
    os.mkdir(base_dir)

if not os.path.exists(docs_dir):
    os.mkdir(docs_dir)

# write first part to pdf
pdf = fpdf.FPDF()
pdf.add_page()
pdf.set_font("Times", size=12)
pdf.write(text= str(part1.encode('utf-8')))
pdf.output(pdf_path)
print(f"saved pdf to {pdf_path}")

# write second part to docx
doc = docx.Document()
doc.add_paragraph(part2)
doc.save(docx_path)
print(f"saved docx to {docx_path}")

# write third part to txt
with open(txt_path, 'w', encoding='utf-8') as txt:
    txt.write(part3)
print(f"saved txt to {txt_path}")

# write qna infos to json file
with open(f"{base_dir}/{questions_file}", 'w', encoding='utf-8') as jf:
    jf.write(json.dumps(qna_infos))
print(f"saved qnas to {base_dir}/{questions_file}")
