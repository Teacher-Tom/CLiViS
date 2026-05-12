import json
import time

from clivis import utils
from clivis.graph import SceneGraph
from clivis.video import spilit_video
from clivis.memory.time_working_memory import TimeWorkingMemory
from clivis.models.vlm import basic_vlm_chat
from clivis.models import llm as chat_llm_only
from clivis.models.llm import basic_llm_chat
from clivis.pipeline.time_utils import (
    DEFAULT_MAX_FPS,
    calculate_period_fps,
    calculate_video_segments,
    extract_question,
    is_valid_period_format,
)


def init_periods_and_graph(
    video_path,
    question,
    fps=1,
    max_pixels=360 * 480,
    output_path="output_segments",
):
    """
    Initialize time segments and the scene graph.
    """
    # Get video duration (seconds).
    video_duration = spilit_video.get_video_duration(video_path)
    print("Video duration: ", video_duration)
    # Convert to mm:ss format.
    video_duration_hhmmss = f"{int(video_duration // 60):02d}:{int(video_duration % 60):02d}"
    # Split into time ranges.
    period_names = calculate_video_segments(video_duration)
    print("Segments:", period_names)
    # Split the video file into segment files.
    segments_to_files = spilit_video.split_video(video_path, period_names, output_path)
    # Call the LLM to extract focus keywords from the question.
    focus_prompt = f"""
Here's a question about a video: {question}
Don't answer the question. Please extract and list all possible keywords related to the question. Output format: [keyword1, keyword2, ...].
    """.strip()
    llm_msgs = [
        {
            "role": "user",
            "content": focus_prompt
        },
    ]
    # Call the LLM to get the question focus.
    focus_response = basic_llm_chat(llm_msgs, temperature=0.3, top_p=0.9, max_tokens=512)

    print(focus_response)

    # Helper to parse a relevance answer.
    def _parse_related(ans: str) -> bool:
        if not isinstance(ans, str):
            return False
        s = ans.strip().lower()
        # Prefer negative matches first.
        neg = ["no", "not related", "irrelevant", "无关", "不相关", "不是"]
        if any(k in s for k in neg):
            return False
        # Then check positive matches, while avoiding false positives from "not".
        if "不是" in s:
            return False
        pos = ["yes", "related", "relevant", "有关", "相关", "是"]
        if any(k in s for k in pos):
            return True
        # Default conservatively to not related.
        return False

    # Generate a description for each segment (after a relevance check).
    periods = []
    max_seg_fps = DEFAULT_MAX_FPS
    for i, (period_name, segment_file) in enumerate(segments_to_files.items()):
        period_info = {}
        period_info["start_time"] = period_name.split("-")[0]
        period_info["end_time"] = period_name.split("-")[1]

        # First ask whether this segment is relevant (expect yes/no only).
        # If there is only one segment, treat it as related.
        if len(segments_to_files) == 1:
            is_related = True
            print(f"[{period_name}] only one segment, directly set as related.")
        else:
            relevance_prompt = f"""Question: {question}
    Is this video segment relevant to the question? Answer only "yes" or "no"."""
            relevance_msgs = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": segment_file,
                            "fps": min(min(4, len(period_name)) * fps, max_seg_fps),
                        },
                        {"type": "text", "text": relevance_prompt},
                    ],
                }
            ]
            relevance_answer = basic_vlm_chat(relevance_msgs, temperature=0.1, top_p=0.8, top_k=10, max_tokens=8)
            is_related = _parse_related(relevance_answer)
            print(f"[{period_name}] relevance: {relevance_answer} -> {is_related}")

        # If not related, skip detailed description.
        if not is_related:
            period_info["description"] = "(Possibly not related)"
            periods.append(period_info)
            continue

        # If related, generate action + scene/object description.
        action_msg = f"Possibly related key words: {focus_response}\nList all people's actions in sequence briefly."
        vlm_msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": segment_file,
                        "fps": min(min(4, len(period_name)) * fps, max_seg_fps),
                    },
                    {"type": "text", "text": action_msg},
                ],
            }
        ]

        period_description_action = basic_vlm_chat(vlm_msgs, temperature=0.1, top_p=0.8, top_k=20, max_tokens=1024)

        # For the last segment, also describe the last frame (useful for action prediction questions).
        end_description = ""
        if i == len(segments_to_files) - 1:
            last_frame_msg = (
                f"Possibly related key words: {focus_response}\n"
                "What does this POV video end with in the last frame? Focus on person's state and actions."
            )
            vlm_msgs = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": segment_file,
                            "fps": min(min(4, len(period_name)) * fps, max_seg_fps),
                        },
                        {"type": "text", "text": last_frame_msg},
                    ],
                },
            ]
            end_description = basic_vlm_chat(
                vlm_msgs, temperature=0.1, top_p=0.8, top_k=20, max_tokens=1024
            )

        vlm_msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": segment_file,
                        "fps": min(min(4, len(period_name)) * fps, max_seg_fps),
                    },
                    {"type": "text", "text": action_msg},
                ],
            },
            {"role": "assistant", "content": period_description_action},
            {
                "role": "user",
                "content": (
                    f"Possibly related key words: {focus_response}\n"
                    "Briefly describe the scene and objects in this POV video. "
                    "Focus on the objects where the person interact."
                ),
            },
        ]
        period_description_objects = basic_vlm_chat(
            vlm_msgs, temperature=0.1, top_p=0.8, top_k=20, max_tokens=1024
        )

        if end_description != "":
            period_description = (
                f"Action: {period_description_action}\n"
                f"Details: {period_description_objects}\n"
                f"Last frame: {end_description}"
            )
        else:
            period_description = f"Action: {period_description_action}\nDetails: {period_description_objects}"

        period_info["description"] = period_description
        periods.append(period_info)

    json_dict = {"periods": periods}
    # Convert json_dict to a string.
    converted_division = json.dumps(json_dict, indent=0)
    print(converted_division)
    # Initialize the graph.
    graph = SceneGraph(json_dict, video_path, output_path, video_duration)
    graph.init_persons_and_areas(converted_division)
    graph.init_obj(converted_division, extract_question(question), video_path)
    graph.init_obj_rel(converted_division)
    graph.init_action(converted_division)
    return graph, converted_division, video_duration, video_duration_hhmmss, focus_response

def initial_analyse_and_mark_key_nodes(graph: SceneGraph, description, question):
    """
    First-pass analysis of the question and marking key nodes.
    :param graph: Scene graph
    :param description: Video segment description
    :param question: Input question
    :return: None
    """
    prompt = """
# Video Segments Description  
{description}  

# Question  
{question}  

# Task Definition  
The above text describes a POV video segment, and there is also a question related to this video. You will need to further interact with the VLM to obtain more information to answer this question. But first, you must analyze the existing description and the question, then output the key entity names you believe are relevant to answering the question.  

## Entity List  
You must select entity names from the following list:  
Characters: {chars}  
Areas: {areas}  
Objects: {objs}  

## Output Format  
Question analysis: (your analysis)  
```json  
{  
    "key_entity_names": ["name1", "name2", ...]  
}  
```
    """.strip()
    prompt = prompt.replace("{description}", description)
    prompt = prompt.replace("{chars}", graph.get_all_char_infos())
    prompt = prompt.replace("{areas}", str(graph.navigation_graph.area_names))
    prompt = prompt.replace("{objs}", str(graph.navigation_graph.obj_names))
    prompt = prompt.replace("{question}", question)
    system_input = "You are a video scene reasoning question answering assistant, and your task is to guide a visual language model (VLM) capable of understanding video to solve complex video reasoning problems step by step. The VLM can watch the video and give a response based on your instructions, and then you can gather reasoning evidence that can help solve the problem and give instructions for next steps. Because VLM can only see a fixed video, he can't zoom in or out."
    msg = [{"role": "system", "content": system_input},
        {"role": "user", "content": prompt}
    ]
    key_entity_names = []
    # Call LLM to output JSON and parse it; retry up to 5 times on failure.
    for i in range(5):
        try:
            response = basic_llm_chat(msg)
            print(response)
            response_dict = utils.parse_json_text(response)
            if response_dict and "key_entity_names" in response_dict:
                key_entity_names = response_dict["key_entity_names"]
                msg.append({"role": "assistant", "content": response})
                break
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            continue
    graph.add_marked_entities(key_entity_names)
    return key_entity_names, msg

def llm_generate_instruction(
    llm_msgs: list,
    vlm_response,
    last_llm_instruction,
    nav_periods_info: str,
    clue_subgraph: str,
    memory_info: str,
    graph: SceneGraph,
    video_duration: str,
    focus: str,
):
    """
    Use the LLM to generate the next instruction for the VLM.
    :param llm_msgs: LLM messages
    :param vlm_response: Last VLM response
    :param nav_periods_info: Navigation graph info
    :param clue_subgraph: Clue subgraph
    :param memory_info: Memory info
    :return: LLM instruction dict
    """
    prompt = """
{nav_periods_info}
**Clue subgraph info:**
{clue_subgraph}

---
The preliminary conclusion of VLM:
{focus}

---
{mem_info}

---
Your last instruction: {instruction}
VLM's response: {vlm_response}

---
Please analyze in combination with the preliminary conclusion of VLM and the video description information.
The above information is the video information collected by VLM so far, and there may still be missing information and errors. You need to keep gathering more information until you can answer the question.
If you need more information, please provide the next instruction to the VLM, including: the time period to focus on next, the information that VLM needs to obtain from that period, etc.
You can specify that information should be fetched from the "full video" or from a certain period of the video. "Full video" is usually chosen only when it is uncertain from which time period to obtain information.
If you don't know in which period to get the information you need, you should let the VLM get global information from the "full video" first. Then you should specify a time period to ask for it, as it provides more details.
The same thing may appear in more than one time period.
VLM can perform basic visual tasks. Sometimes VLM may also make mistakes.
You should only give a simple and short instruction or question at a time, including recognition, localization, judgment, and description of basic scenes, objects and actions. Ask one thing at a time.
For example: 
- "Where's xx?"
- "What is the person doing?"
- "Describe xx."
- "Describe the time period in detail."
- "What does this video end with?"
**DO NOT** ask Yes/No questions
For next action prediction questions, you need to pay additional attention to person's action or state at the end of the video. VLM cannot see what happens after the last frame. So infer the subsequent actions based on the character status in the last frame and the previous information.
Total video duration: {video_duration}
You should conduct the analysis first:
Analysis: The analysis of the current situation and info. Determine whether the existing information can lead to an answer. Otherwise, you should give an instruction for missing info.

Then give the instruction or answer in json form:
Your instructions should follow the following format:
```json
{
    "type": "scene/object/action",
    "period": "only select "full video" or the following time periods(hh:mm:ss-hh:mm:ss): {period_names}",
    "instruction": "A one-sentence question to VLM less than 20 words."
}
```
IMPORTANT: DO NOT repeat previous instructions. Ask only one thing at one time.

Only if current information is sufficient to arrive at the answer, give the final answer.
If some of the information is still missing after multiple attempts, please give the closest answer.
If it is a multiple-choice question, the answer must include option symbols.
The answer format is as follows:
```json
{
    "reason": "The detailed reasoning process and rationales. Indicate the corresponding time period if necessary.",
    "answer": "Only one closest answer to the question. 'No answer' shouldn't be given."
}
```
    """
    fake_prompt = """
Your last instruction: {instruction}
VLM's last response: {vlm_response}

---
The above information is the video information collected by VLM so far, and there may still be missing information and errors. You need to keep gathering more information until you can answer the question.
If you need more information, please provide the next instruction to the VLM, including: the time period to focus on next, the information that VLM needs to obtain from that period, etc.
You can specify that information should be fetched from the "full video" or from a certain period of the video. "Full video" is usually chosen only when it is uncertain from which time period to obtain information.
If you don't know in which period to get the information you need, you should let the VLM get global information from the "full video" first. Then you should specify a time period to ask for it, as it provides more details.
The same thing may appear in more than one time period.
VLM can perform basic visual tasks. Sometimes VLM may also make mistakes.
You should only give a simple and short instruction or question at a time, including recognition, localization, judgment, and description of basic scenes, objects and actions. Ask one thing at a time.
IMPORTANT: DO NOT repeat previous instructions, especially if VLM already responded to them.
You should conduct the analysis first:
Analysis: The analysis of the current situation and info. Determine whether the existing information can lead to an answer. Otherwise, you should give an instruction for missing info.

Then give the instruction or answer in json form:
Your instructions should follow the following format:
```json
{
    "type": "scene/object/action",
    "period": "only select "full video" or the following time periods(hh:mm:ss-hh:mm:ss): {period_names}",
    "instruction": "A one-sentence question to VLM less than 20 words."
}

**The final answer should be output immediately if the information and rationales are enough.**
If some of the information is still missing after multiple attempts, please give the closest answer.
The answer format is as follows:
```json
{
    "reason": "The detailed reasoning process and rationales. Indicate the corresponding time period if necessary.",
    "answer": "The closest answer to the question. 'No answer' shouldn't be given."
}
```
    """
    period_names = str(graph.navigation_graph.get_period_names())
    prompt = (
        prompt.replace("{nav_periods_info}", nav_periods_info)
        .replace("{clue_subgraph}", clue_subgraph)
        .replace("{vlm_response}", vlm_response)
        .replace("{mem_info}", memory_info)
        .replace("{period_names}", period_names)
        .replace("{instruction}", last_llm_instruction)
        .replace("{focus}", focus)
        .replace("{video_duration}", video_duration)
        .strip()
    )
    fake_prompt = (
        fake_prompt.replace("{vlm_response}", vlm_response)
        .replace("{instruction}", last_llm_instruction)
        .replace("{focus}", focus)
        .replace("{period_names}", period_names)
        .strip()
    )
    llm_msgs.append({"role": "user", "content": prompt})
    # Call LLM to output JSON and parse it; retry up to 5 times on failure.
    for i in range(5):
        try:
            response = chat_llm_only.basic_llm_chat(llm_msgs[-17:], temperature=0.7, top_p=0.9)
            print(response)
            response_dict = utils.parse_json_text(response)
            if "instruction" in response_dict and "period" in response_dict:
                # Validate period format.
                if not is_valid_period_format(response_dict["period"]):
                    print(f"Invalid period format: {response_dict['period']}")
                    continue
                llm_msgs.pop()
                llm_msgs.append({"role": "user", "content": fake_prompt})
                llm_msgs.append({"role": "assistant", "content": response})
                return response_dict
            if "answer" in response_dict and "reason" in response_dict:
                llm_msgs.pop()
                llm_msgs.append({"role": "user", "content": fake_prompt})
                llm_msgs.append({"role": "assistant", "content": response})
                return response_dict
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            continue
    return None
def inference(video_path, question, output_path="output_segments", fps=1, max_pixels=360 * 480, max_rounds=15):
    start_time = time.time()
    round_count = 0
    triples_count_turn = [] # Track the number of triples per round.
    rationales_count_turn = [] # Track the number of rationales/evidence items per round.

    # Initialize the graph and memory.
    graph, description, video_duration, video_duration_hhmmss, focus = init_periods_and_graph(video_path, question, fps, max_pixels, output_path)
    key_entity_names, llm_msgs = initial_analyse_and_mark_key_nodes(graph, description, question)
    memory = TimeWorkingMemory(question, [])
    # Call VLM to list information/evidence related to the question.
    evidence_prompt = f"{question}\nAbove is a question related to the POV video. Please break down the video content and list all possible info, evidence or clues related to the question. **Don't give the answer.** Let's think step by step."
    vlm_msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": max_pixels,
                    "fps": fps,
                },
                {"type": "text", "text": evidence_prompt},
            ],
        },
    ]
    # Call VLM.
    evidence = basic_vlm_chat(vlm_msgs, temperature=0.1, top_p=0.9, top_k=50, max_tokens=1024)
    print("evidence: ", evidence)
    init_time = time.time()
    print("Initial MAP time: ", init_time - start_time)
    vlm_msgs.pop()
    vlm_msgs.append({"role": "user", "content": evidence_prompt})
    vlm_msgs.append({"role": "assistant", "content": evidence})
    triples_count_turn.append(graph.relation_graph.count_triples())

    # Iterative reasoning.
    vlm_response = "None"
    llm_instruction = "None"
    # vlm_msgs = []
    for i in range(max_rounds):
        round_count += 1
        print("-----------------------------------")
        print(f"Round {i + 1}")
        # Iterative reasoning.
        # Output navigation graph info.
        nav_periods_info = graph.navigation_graph.output_periods_info()
        # print("nav_preiods_info:", nav_periods_info)
        # Output clue subgraph.
        clue_subgraph = graph.output_clue_subgraph(list(graph.marked_entity_names), 10)
        print("clue_subgraph:", clue_subgraph)
        # Output memory info.
        memory_info = memory.output_memory_info()
        # Call LLM to generate the next instruction.
        response_dict = llm_generate_instruction(
            llm_msgs=llm_msgs,
            vlm_response=vlm_response,
            last_llm_instruction=llm_instruction,
            nav_periods_info=nav_periods_info,
            clue_subgraph=clue_subgraph,
            memory_info=memory_info,
            graph=graph,
            video_duration=video_duration_hhmmss,
            focus=evidence,
        )
        period = "full video"
        instruction = None
        if response_dict and "reason" in response_dict and "answer" in response_dict:
            elapsed_time = time.time() - start_time
            return json.dumps(response_dict), {"round_count": round_count, "elapsed_time": elapsed_time, "triples_count_turn": triples_count_turn, "video_duration": video_duration_hhmmss, "rationales_count_turn": rationales_count_turn}
        if response_dict and "period" in response_dict and "instruction" in response_dict:
            period = response_dict["period"]
            instruction = response_dict["instruction"]
        else:
            print("LLM fail to generate instruction, exit")
            break

        # Validate/resolve the requested period.

        if period != "full video" and period not in graph.navigation_graph.get_period_names():
            matched_periods = spilit_video.find_time_range(period, graph.navigation_graph.get_period_names())
            if len(matched_periods) > 1:
                time_locate_result = f"According to full video, the time periods that may be relevant to your directive are: {matched_periods}. You can ask VLM to focus on them."
            if len(matched_periods) > 0:
                period = matched_periods[0]
            else:
                print(f"Period '{period}' not in period_names")
        # Select the video path to query.
        if period not in graph.navigation_graph.get_period_names():
            focus_segment_path = video_path
        else:
            focus_segment_path = graph.get_vid_period_segment_path(period)
        # Call VLM for a response.
        period_fps = fps
        if period not in graph.navigation_graph.get_period_names():
            llm_instruction = f'''{instruction}\nAbove is a question related to the video. Please think step by step and provide a detailed answer with reason.'''
            vlm_msgs.append({
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": focus_segment_path,
                        # "max_pixels": max_pixels,
                        "fps": period_fps,
                    },
                    {"type": "text", "text": llm_instruction},
                ],
            })
            input_vlm_msgs = [
                                 # {"role": "system",
                                 #  "content": f"This is a POV video. Possibly related key words: {focus}\n"},
                                 # {"role": "user", "content": f"Summarize the keywords of the video."},
                                 # {"role": "assistant", "content": focus},
                                 # {
                                 #     "role": "user",
                                 #     "content": f"""Please divide the video into multiple periods based on the video content. And describe each period briefly.""".strip()
                                 # },
                                 # {
                                 #     "role": "assistant",
                                 #     "content": graph.navigation_graph.output_periods_description()
                                 # },
                             ] + vlm_msgs[-7:]


        else:
            # llm_instruction = f'''This video is a snippet({period}) of the full video. Please answer the question only based on the snippet: {instruction}'''
            llm_instruction = f'''{instruction}\nAbove is a question related to the video. Please think step by step and provide a detailed answer with reason.'''
            # Compute FPS based on period length.
            period_fps = calculate_period_fps(period, fps, video_duration)
            vlm_msgs.append({
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": focus_segment_path,
                        # "max_pixels": max_pixels,
                        "fps": period_fps,
                    },
                    {"type": "text", "text": llm_instruction},
                ],
            })
            input_vlm_msgs = [
                                 # {"role": "system",
                                 #  "content": f"This is a POV video. Possibly related key words: {focus}\n"},
                                 # {"role": "user", "content": f"Summarize the keywords of the video."},
                                 # {"role": "assistant", "content": focus},
                                 # {
                                 #     "role": "user",
                                 #     "content": f"""Please divide the video into multiple periods based on the video content. And describe each period briefly.""".strip()
                                 # },
                                 # {
                                 #     "role": "assistant",
                                 #     "content": graph.navigation_graph.output_periods_description()
                                 # }
                             ] + vlm_msgs[-7:]

        # input_vlm_msgs = vlm_msgs[-7:]
        for k in range(5):
            try:
                vlm_response = basic_vlm_chat(input_vlm_msgs, temperature=0.1, top_p=0.8, max_tokens=4096)
                # print("VLM response: ", vlm_response)
                if "bbox" in vlm_response:
                    raise ValueError("Invalid response format")
                break
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                continue
        vlm_msgs.pop()
        vlm_msgs.append({"role": "user", "content": llm_instruction})
        vlm_msgs.append({"role": "assistant", "content": vlm_response})
        vlm_response = f"({period}) " + vlm_response
        print(vlm_response)
        memory.update_history_msg(llm_instruction, vlm_response)
        memory.extract_and_update_rationale_list(period)
        rationales_count_turn.append(memory.get_rationale_count())
        if period in graph.navigation_graph.get_period_names():
            graph.update_obj_rel_act(vlm_response, llm_instruction, period, question)
            triples_count_turn.append(graph.relation_graph.count_triples())
    # Exceeded max rounds; ask the LLM for a final answer.
    print("Reach max rounds")
    # Output navigation graph info.
    nav_periods_info = graph.navigation_graph.output_periods_info()
    # Output clue subgraph.
    clue_subgraph = graph.output_clue_subgraph(list(graph.marked_entity_names), 10)
    # Output memory info.
    memory_info = memory.output_memory_info()

    # Build the final answer prompt.
    llm_input = """
{nav_periods_info}
**Clue subgraph info:**
{clue_subgraph}

---
{mem_info}

---
Based on the above information and previous messages, reason and answer the question. Even if some information is missing, please give the closest answer.
""".strip()
    llm_input = llm_input.replace("{nav_periods_info}", nav_periods_info).replace("{clue_subgraph}", clue_subgraph).replace("{mem_info}", memory_info).strip()
    llm_msgs.append({"role": "user", "content": llm_input})
    final_response = basic_llm_chat(llm_msgs)
    print("Final response: ", final_response)
    elapsed_time = time.time() - start_time
    return final_response, {"round_count": round_count, "elapsed_time": elapsed_time, "triples_count_turn": triples_count_turn, "video_duration": video_duration_hhmmss, "rationales_count_turn": rationales_count_turn}




