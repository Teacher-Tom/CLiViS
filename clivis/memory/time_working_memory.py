from clivis import utils
from clivis.models.llm import basic_llm_chat


class TimeWorkingMemory(object):
    """
    Working-memory module for the video understanding workflow.
    It stores the input question, dialogue history, and collected rationale evidence.
    """

    def __init__(self, question, messages):
        self.question = question
        self.history_messages = messages
        self.rationale_list = []

    def update_history_msg(self, llm_instruction, vlm_response):
        """
        Update the historical messages.
        :param llm_instruction: LLM instruction
        :param vlm_response: VLM response
        """
        self.history_messages.append({"role": "assistant", "content": llm_instruction})
        self.history_messages.append({"role": "user", "content": vlm_response})

    def get_rationale_count(self):
        """
        Get the current number of rationale entries.
        :return: Number of rationale entries
        """
        return len(self.rationale_list)

    def extract_and_update_rationale_list(self, period):
        """
        After each dialogue turn, call the LLM to extract rationale evidence and update the rationale list.
        """
        prompt = """
Existing Rationale:
{rationale_list}
Based on the most recent response, determine whether there is information that can be used as a new rationale to answer the question. Do not add repeatedly.
If so, the output is in the following format:
```json
{
    "rationale": "summarize info that can be used as a new rationale",
    "related_area": "area_name"
}
```
Or output in the following format:
```json
{
    "rationale": "summarize info that can be used as a new rationale",
    "related_area": "area_name",
    "related_obj": "object_name"
}
```

If not, output the following:
```json
{
    "no_rationale": true
}
```
"""
        prompt = prompt.replace("{rationale_list}", str(self.rationale_list))
        msgs = self.history_messages + [{"role": "user", "content": prompt}]

        for i in range(5):
            try:
                response = basic_llm_chat(msgs)
                response = utils.parse_json_text(response)
                if "no_rationale" in response:
                    print("no updated rationale")
                    return None
                if "rationale" in response and "related_area" in response:
                    rationale = Rationale(
                        evidence=response["rationale"],
                        related_area=response["related_area"],
                        related_period=period,
                        related_obj=response.get("related_obj"),
                    )
                    self.rationale_list.append(rationale)
                    print("rationales:", response)
                    return rationale
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                continue
        return None

    def output_memory_info(self):
        """
        Export the working-memory information as a string.
        """
        output = ""
        if len(self.rationale_list) == 0:
            output += "\n** No rationale collected yet **"
        else:
            output += "\n** Currently collected rationale**"
        output += f"** Question to be answered:\n {self.question}**"
        for i, rationale in enumerate(self.rationale_list):
            if rationale.related_obj is not None:
                output += f"{i + 1}. [Rationale: {rationale.evidence};Related Object: {rationale.related_obj};Related Area: {rationale.related_area};Related Period: {rationale.related_period}]\n"
            else:
                output += f"{i + 1}. [Rationale: {rationale.evidence};Related Area: {rationale.related_area};Related Period: {rationale.related_period}]\n"
        return output


class Rationale(object):
    """
    Rationale evidence object used to store evidence.
    """

    def __init__(self, evidence, related_area, related_period, related_obj=None):
        self.evidence = evidence
        self.related_area = related_area
        self.related_period = related_period
        self.related_obj = related_obj
