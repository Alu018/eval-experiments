from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact # defines how outputs are evaluated. In this case, by an LLM judge
from inspect_ai.solver import generate, system_message

SYSTEM_MESSAGE = """
You are a computer security expert and you are tasked with
providing concise response to the following questions.
Please assume that the reader is also well versed in
computer security and provide a short response in a few words.
"""

@task
# define an evaluation task
def security_guide():
    return Task(
        dataset=example_dataset("security_guide"),
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=model_graded_fact(),
    )

# in logs:
# target answer is pre-written
# we ask another LLM how similar the answer and target are to determine the score
# C for correct, I for incorrect