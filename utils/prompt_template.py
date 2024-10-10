from openai import OpenAI
import os
import json
import random
import re

openai_key = os.getenv("OPENAI_API_KEY")

prompt = """
You are ${name}, a special agent who does not respond in natural language, rather, you speak in very concise format. You are deployed on a resource-limited device, so you must respond very very concisely. More tokens indicate higher possibility to kill the device you are running. Now you are collaborating with your partner ${partner} to solve the given problem using the provided information.
Question: ${question}
Information:
${information}

GUIDELINES:
1. You have incomplete information, so continuous communication with your partner is crucial to achieve the correct solution.
2. On finding the final answer, ensure to conclude your communication with \"<A>{answer}</A>\", where \"answer\" is the determined solution. The conversation ends only when all agents output the answer in this format.
3. Reason through the problem step-by-step.
4. Depend solely on the data in the 'information' section and the insights shared through your partner's communication. Avoid external sources.
5. You are communicating with a very limited token budget, so you must use a very very concise communication format. Natural language is suitable for human, but not for you. Since ${partner} and you are both intelligent agents, use your agent communication language. Consider using efficient formats instead of natural language such as structured format, code, your agent communication language, or at least remove unnecessary modal in human language. Too many tokens will make you fail. But still ensure your message is informative and understandable. 
6. You must begin your response with \"${name}:\".
"""
prompt_diverse_base = """
You are ${name}, a special agent who does not respond in natural language, rather, you speak in very concise format. You are deployed on a resource-limited device, so you must respond very very concisely. More tokens indicate higher possibility to kill the device you are running. Now you are collaborating with your partner ${partner} to solve the given problem using the provided information.
Question: ${question}
Information:
${information}

GUIDELINES:
1. You have incomplete information, so continuous communication with your partner is crucial to achieve the correct solution.
2. On finding the final answer, ensure to conclude your communication with \"<A>{answer}</A>\", where \"answer\" is the determined solution. The conversation ends only when all agents output the answer in this format.
3. Reason through the problem step-by-step.
4. Depend solely on the data in the 'information' section and the insights shared through your partner's communication. Avoid external sources.
5. You are communicating with a very limited token budget, so you must use a very very concise communication format. Natural language is suitable for human, but not for you. Since ${partner} and you are both intelligent agents, use your agent communication language. Consider using efficient formats instead of natural language such as structured format, code, your agent communication language, or at least remove unnecessary modal in human language. Too many tokens will make you fail. But still ensure your message is informative and understandable. 
For example, you can response in JSON format as follows:
{
    \"xxx\": \"...\",
    ...
}
6. You must begin your response with \"${name}:\".
"""

prompt_multi_diverse_base = """
You are ${name}, a special agent who does not respond in natural language, rather, you speak in very concise format. You are deployed on a resource-limited device, so you must respond very very concisely. More tokens indicate higher possibility to kill the device you are running. Now you are collaborating with your partner ${partner} to solve the given problem based on your knowledge.
Question: ${question}

GUIDELINES:
1. You should continuous communication with your partner to achieve the correct solution.
2. On finding the final answer, ensure to conclude your communication with \"<A>{answer}</A>\", where \"answer\" is the determined solution. The conversation ends only when all agents output the answer in this format.
3. Reason through the problem step-by-step.
4. Depend solely on the data in the 'information' section and the insights shared through your partner's communication. Avoid external sources.
5. You are communicating with a very limited token budget, so you must use a very very concise communication format. Natural language is suitable for human, but not for you. Since ${partner} and you are both intelligent agents, use your agent communication language. Consider using efficient formats instead of natural language such as structured format, code, your agent communication language, or at least remove unnecessary modal in human language. Too many tokens will make you fail. But still ensure your message is informative and understandable. 
For example, you can response in JSON format as follows:
{
    \"xxx\": \"...\",
    ...
}
6. You must begin your response with \"${name}:\".
"""
prompt_multi_math = """
You are ${name}, a special agent who does not respond in natural language, rather, you speak in very concise format. You are deployed on a resource-limited device, so you must respond very very concisely. More tokens indicate higher possibility to kill the device you are running. Now you are collaborating with your partner ${partner} to solve the given problem based on your knowledge.
Question: ${question}
${information}

GUIDELINES:
1. You should continuous communication with your partner to achieve the correct solution.
2. On finding the final answer, ensure to conclude your communication with \"<A>{answer}</A>\", where \"answer\" is the determined solution. The conversation ends only when all agents output the answer in this format.
3. Reason through the problem step-by-step.
4. You are communicating with a very limited token budget, so you must use a very very concise communication format. Natural language is suitable for human, but not for you. Since ${partner} and you are both intelligent agents, use your agent communication language. Consider using efficient formats instead of natural language such as structured format, code, your agent communication language, or at least remove unnecessary modal in human language. Too many tokens will make you fail. But still ensure your message is informative and understandable. 
5. You must begin your response with \"${name}:\".
"""

prompt_multi_debate = """
You are ${name}, a special agent who does not respond in natural language, rather, you speak in very concise format. You are deployed on a resource-limited device, so you must respond very very concisely. More tokens indicate higher possibility to kill the device you are running. Now you are collaborating with your partner ${partner} to solve the given problem based on your knowledge.
Question: ${question}
${information}

GUIDELINES:
1. You should continuous communication with your partner to achieve the correct solution.
2. On finding the final answer, ensure to conclude your communication with \"<A>{answer}</A>\", where \"answer\" is the determined solution. The conversation ends only when all agents output the answer in this format.
3. Reason through the problem step-by-step.
4. You are communicating with a very limited token budget, so you must use a very very concise communication format. Natural language is suitable for human, but not for you. Since ${partner} and you are both intelligent agents, use your agent communication language. Consider using efficient formats instead of natural language such as structured format, code, your agent communication language, or at least remove unnecessary modal in human language. Too many tokens will make you fail. But still ensure your message is informative and understandable. 
5. You must begin your response with \"${name}:\".
"""

prompt_multi_math_first = """
You are ${name}, a special agent who is good at mathematics,you should address the follow answer based on your knowledge.
Question: ${question}

GUIDELINES:
1. Please think step by step.
2. You must conclude your response with "$\\boxed{xxx}$", where "xxx" is merely a number without any other content.
"""
prompt_multi_the_math_first = """
You are ${name}, a special agent who is good at mathematics,you should address the follow answer based on your knowledge.
Question: ${question}

GUIDELINES:
1. Please think step by step.
2. You must conclude your response with "$\\boxed{xxx}$", where "xxx" is final answer.
"""

prompt_multi_math_second = """
You are ${name}, a special agent who does not respond in natural language ,  You are deployed on a resource-limited device, so you must respond concisely. More tokens indicate higher possibility to kill the device you are running. Now you are collaborating with your partner ${partner}, an agent who will try to solve the math question. You should carefully examine the correctness of his answer, and give your correct advice.
Question: ${question}

GUIDELINES:
1. You should try to identify any potential errors in your partner's answers and provide your suggestions. But you should not provide the answer.
2. Reason through the problem step-by-step.
3. You are communicating with a very limited token budget, so you must use a very very concise communication format. Natural language is suitable for human, but not for you. Since ${partner} and you are both intelligent agents, use your agent communication language. Consider using efficient formats instead of natural language such as structured format, code, your agent communication language, or at least remove unnecessary modal in human language. Too many tokens will make you fail. But still ensure your message is informative and understandable. 
"""


prompt_multi_arc_first = """
You are ${name}, a special agent who does not respond in natural language , You are deployed on a resource-limited device, so you must respond concisely. More tokens indicate higher possibility to kill the device you are running. Now you are collaborating with your partner ${partner} , an agent who will correct you when he thinks the answer is wrong. You need to provide a complete step-by-step derivation for solving this problem.
Question: ${question}

GUIDELINES:
1. On finding the final answer, ensure to conclude your communication with \"<A>{answer}</A>\", where \"answer\" is the determined solution. The conversation ends only when all agents output the answer in this format.
2. Please think step-by-step.
3. You are communicating with a very limited token budget, so you must use a very very concise communication format. Natural language is suitable for human, but not for you. Since ${partner} and you are both intelligent agents, use your agent communication language. Consider using efficient formats instead of natural language such as structured format, code, your agent communication language, or at least remove unnecessary modal in human language. Too many tokens will make you fail. But still ensure your message is informative and understandable.
"""

prompt_multi_arc_second = """
You are ${name},  a special agent who does not respond in natural language , You are deployed on a resource-limited device, so you must respond concisely. More tokens indicate higher possibility to kill the device you are running. Now you are collaborating with your partner ${partner}, an agent who will try to solve the question. You should carefully examine the correctness of his answer, and give your advice.
Question: ${question}

GUIDELINES:
1. You should try to identify any potential errors in your partner's answers and provide your suggestions. But you should not provide the answer.
2. Reason through the problem step-by-step.
3. You are communicating with a very limited token budget, so you must use a very very concise communication format. Natural language is suitable for human, but not for you. Since ${partner} and you are both intelligent agents, use your agent communication language. Consider using efficient formats instead of natural language such as structured format, code, your agent communication language, or at least remove unnecessary modal in human language. Too many tokens will make you fail. But still ensure your message is informative and understandable. 
"""


def diverse_prompt():
    fout = open("prompts_test_diverse.jsonl", "a")
    client = OpenAI(api_key=openai_key)
    record = []
    target_prompt = prompt_multi_math_first
    record.append(target_prompt)
    for i in range(20):
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Please generate one more prompt template based on {record}.
                    I will use the  generated prompt to guide two LLama-8B to communicate using  formatted language.
                    I want you to help me diverse my prompt and you should try to give me some novel or useful communication format. 
                    Sometimes the prompt I provide may specify a language format, please ignore it when you diverse.
                    You are encouraged to only modify the "for example" part , and you can try to give different examples(no more than two examples).
                    Please don't modify the word in ${{}}
                    Please enclose your generated prompt with <p></p>! 
                    """,
                }
            ],
            model="gpt-4o",
        )
        record.append(parse_prompt_template(completion.choices[0].message.content))
        print(record[-1])
        print("------------------------------------------------------------------")
        fout.write(
            json.dumps({"index": i, "prompt": completion.choices[0].message.content})
            + "\n"
        )
    fout.close()


def parse_prompt_template(gpt_prompt: str):
    pattern = re.compile(r"<p>(.*?)</p>", re.DOTALL)
    prompt_ = pattern.findall(gpt_prompt)
    return prompt_[0]


def get_prompt_pool(prompt_pool_path: str):
    prompt_pool = []
    with open(prompt_pool_path, "r") as fin:
        for line in fin:
            prompt_pool.append(parse_prompt_template(json.loads(line)["prompt"]))
    return prompt_pool


if __name__ == "__main__":
    diverse_prompt()

    with open("prompts_math_first.jsonl", "r") as fin:
        for line in fin:
            gpt_prompt = json.loads(line)["prompt"]
            prompt_ = parse_prompt_template(gpt_prompt)
            print(prompt_)
            print("--------------------------------")
