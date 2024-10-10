import json


def lines_to_list(input_path: str, output_path: str):
    last_question = None
    result_list = []
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            result = data["results"][0]
            if last_question is None or result["question"] != last_question:
                if last_question is not None:
                    with open(output_path, "a") as fout:
                        fout.write(json.dumps({"results": result_list}) + "\n")
                last_question = result["question"]
                result_list = []
            result_list.append(result)
    with open(output_path, "a") as fout:
        fout.write(json.dumps({"results": result_list}) + "\n")
