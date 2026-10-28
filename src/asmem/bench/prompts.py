"""Prompts for the benchmark pipeline: answer, answer-judge, rubric-judge.

The prompts are deliberately short. Every field is filled with ``str.replace``
rather than ``str.format`` so that YAML overrides can contain JSON braces
without escaping them.
"""

from __future__ import annotations

ANSWER_SYSTEM = """You are a careful question answerer. Answer from the evidence below. \
Use only what the evidence supports; if it does not contain the answer, say so instead of guessing. \
Before concluding that the evidence lacks the answer, scan every record for the requested value: \
a short conversational turn is valid evidence, and you should quote the name, place, date, or \
number it states rather than refusing. \
A line beginning with "[context <id>]" is surrounding context attached to the cited record; \
use it to resolve pronouns and dates, but treat the cited record as the source of the fact. \
A line beginning with "[resolved:" gives the absolute date derived from the record's observed_at \
(for example "yesterday = 2023-05-07"); use that date when the question asks when something happened. \
Do not mention the retrieval process or the evidence format."""

#: ScriptMem ships no source scripts (license), so its questions/options are the
#: only input. This prompt lets the model answer the MCQ itself instead of
#: refusing for lack of retrieved evidence.
SCRIPTMEM_SYSTEM = """You are answering a self-contained multiple-choice question. \
The question and options contain all the information you have. Choose the most defensible \
option(s) and return only the requested letters. Do not explain your reasoning."""

#: PersonaMem gold answers are personalized advice, not a fact copied from a
#: record. The evidence constrains the personalization; general advice is fine.
PERSONALIZED_SYSTEM = """You are a helpful assistant answering the user's request. \
Use the evidence about the user's preferences and history to personalize the answer. \
You may use general knowledge for advice, but do not contradict stored preferences \
and do not invent personal facts."""

ANSWER_USER = """Question:
{question}

Evidence:
{evidence}
{options}
{style}

Answer:"""

#: Dataset-specific answer instructions. They are appended to the user prompt.
ANSWER_STYLES = {
    "longmemeval_oracle": (
        "Answer concisely. If the question asks for a date, name, number, or entity, "
        "give exactly that value; if it asks for an event, name the event."
    ),
    "longmemeval_s": (
        "Answer concisely. If the question asks for a date, name, number, or entity, "
        "give exactly that value; if it asks for an event, name the event."
    ),
    "longmemeval_m": (
        "Answer concisely. If the question asks for a date, name, number, or entity, "
        "give exactly that value; if it asks for an event, name the event."
    ),
    "beam_100k": (
        "Answer as the assistant would in the conversation. Follow any format, length, "
        "or content constraint stated in the question."
    ),
    "beam_500k": (
        "Answer as the assistant would in the conversation. Follow any format, length, "
        "or content constraint stated in the question."
    ),
    "beam_1m": (
        "Answer as the assistant would in the conversation. Follow any format, length, "
        "or content constraint stated in the question."
    ),
    "beam_10m": (
        "Answer as the assistant would in the conversation. Follow any format, length, "
        "or content constraint stated in the question."
    ),
    "personamem_32k": (
        "Answer the user's request helpfully and personally, using the evidence about "
        "their preferences, history, and previous instructions."
    ),
    "personamem_128k": (
        "Answer the user's request helpfully and personally, using the evidence about "
        "their preferences, history, and previous instructions."
    ),
    "clbench": (
        "Complete the task. Follow every formatting, content, and process constraint "
        "stated in the question."
    ),
    "scriptmem": (
        "This is a multiple-choice question. Choose the most defensible option(s) from "
        "the question itself, then return ONLY the option letter(s), enclosed in "
        "parentheses, e.g. (B) for one answer or (A, C, D) for several. "
        "For ordering questions, return the letters in the required order."
    ),
    "locomo_refined": (
        "Answer concisely. For temporal questions, give the date at the granularity "
        "supported by the evidence (day/week/month/year); for list answers, include "
        "all required items and do not add unsupported details. If a record carries "
        "an observed_at/session time, do not report that as the event date; use it "
        "only as the anchor for relative expressions such as 'yesterday' or 'last "
        "year'. When the evidence also carries a '[resolved: ...]' annotation, "
        "include that absolute date if the question asks when something happened."
    ),
}
DEFAULT_ANSWER_STYLE = "Answer the question directly and concisely."


JUDGE_SYSTEM = """You are an exacting grader. Compare the predicted answer with the gold answer \
using the question as context. Accept semantically equivalent answers, but do not reward answers \
that contradict the gold answer or answer a different question."""

JUDGE_USER = """Question:
{question}

Gold answer:
{gold}

Predicted answer:
{predicted}

Return one JSON object:
{{"correct": true or false, "reason": "one short sentence"}}

Rules:
- If the gold answer lists several acceptable alternatives, any one of them is correct.
- Numeric/date values are correct when they denote the same value (e.g. 14 days and two weeks).
- A refusal is correct only when the question is unanswerable from the evidence or the gold is a refusal.
"""


#: Dataset-specific grading notes for the free-form answer judge.
JUDGE_STYLES = {
    "longmemeval_oracle": (
        "The gold answer may list several acceptable alternatives; any one of them is correct."
    ),
    "longmemeval_s": (
        "The gold answer may list several acceptable alternatives; any one of them is correct."
    ),
    "longmemeval_m": (
        "The gold answer may list several acceptable alternatives; any one of them is correct."
    ),
    "personamem_32k": (
        "The gold answer is one acceptable personalized response. Judge whether the predicted "
        "answer is consistent with the user's relevant preference/history and does not "
        "contradict an updated preference; wording may differ."
    ),
    "personamem_128k": (
        "The gold answer is one acceptable personalized response. Judge whether the predicted "
        "answer is consistent with the user's relevant preference/history and does not "
        "contradict an updated preference; wording may differ."
    ),
    "locomo_refined": (
        "Apply the LoCoMo-Refined principle: inclusive without contradiction, complete "
        "without overreach. The prediction must include all required information from one "
        "acceptable gold answer, must not contradict it, must not add unsupported details, "
        "and must preserve the gold's temporal granularity. Any listed acceptable answer "
        "candidate is fully correct; do not require the prediction to combine candidates."
    ),
}


#: Official LoCoMo-Refined judge prompt (src/llm_judge.py in the dataset repo).
#: It asks for a single ``{"label": "CORRECT"|"WRONG"}`` object. The runner
#: calls it once per acceptable gold candidate and accepts any CORRECT.
LOCOMO_REFINED_JUDGE_USER = """Your task is to label an answer as 'CORRECT' or 'WRONG' given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle - Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold's key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT - even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold's content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR<->HOUR, DAY<->DAY, MONTH<->MONTH, YEAR<->YEAR.
  Do not answer a gold at a different time unit - even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
- Do NOT convert relative <-> absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., "the/last/previous/just prior") as equivalent when both the anchor date and the time unit are the same.

Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover ALL of them.
- Extra non-contradictory items generally count as WRONG.
  Example: gold = A, B, C ; gen = A, B, C -> CORRECT
  Example: gold = A, B, C ; gen = A, B, C, D -> WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C -> C, C'), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most"):
- If gold lists multiple reasons/aspects, the generated answer only needs to include ANY ONE of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Just return the label CORRECT or WRONG in a json format with the key as "label":
{"label": "CORRECT" or "WRONG"}
"""


RUBRIC_SYSTEM = """You are grading an answer against a list of rubrics. Decide each rubric \
independently from the answer. Be strict about facts and constraints, but do not require exact wording. \
If a rubric is phrased as a failure condition (for example "Fail if ..."), return true when the answer \
avoids that failure."""

RUBRIC_USER = """Question:
{question}

Answer:
{answer}

Rubrics:
{rubrics}

Return one JSON object:
{{"results": [true, false, ...], "reason": "one short sentence"}}

The results array must have exactly one boolean per rubric, in the same order. \
true means the answer satisfies that rubric; false means it does not.
"""


def answer_style(dataset: str, styles: dict[str, str] | None = None) -> str:
    if styles and dataset in styles:
        return styles[dataset]
    return ANSWER_STYLES.get(dataset, DEFAULT_ANSWER_STYLE)
