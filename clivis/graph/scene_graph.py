"""High-level temporal scene graph orchestration."""

from clivis.graph.common import *
from clivis.graph.navigation_graph import NavigationGraph
from clivis.graph.relation_graph import RelationGraph

class SceneGraph(object):
    def __init__(self, area_time_description, video_path, output_path, video_duration):
        self.navigation_graph = NavigationGraph(area_time_description, video_path, output_path, video_duration)
        self.relation_graph = RelationGraph()
        self.marked_entity_names = set()

    def update_obj_rel_act(self, input_text, llm_instruction, period, question):
        """
        Update the scene graph by adding new entities, relationships, and actions.
        """
        print("update obj rel act")
        # Fetch existing nodes/edges for this time period.
        entity_names = self.navigation_graph.get_entities_in_period(period)
        entity_name_list = []
        for key, value in entity_names.items():
                entity_name_list.extend(value)
        subgraph = self.relation_graph.extract_subgraph_by_nodes(entity_name_list, max_path_length=10)
        # Format the subgraph. (Activities include ordering metadata for later appends.)
        subgraph_json_text = self.relation_graph.format_subgraph_json(subgraph)

        # Fill the prompt for the LLM to propose new entities/relations.
        prompt = """
# Related Question
{question}

# Scene Graph Update Request

## Current Scene Graph Information
{subgraph}

## Input Text
The following text describes a new Q&A with VLM within the period '{period}':
Q: {llm_instruction}
A: {input_text}

## Task
Based on the input text, extract new objects, relationships, and actions that should be added to the current scene graph for this period.

1. For objects: Compare with the existing objects and identify any new objects mentioned in the input text.
2. For relationships: Identify static relationships (not actions) between entities, such as spatial relationships, containment, or attributes.
3. For actions: Identify dynamic interactions where a person performs an action on or with other entities.
Only add new information related to the question when the input text contains it but not in the scene graph. Do not add unnecessary or repetitive information. If there is no useful information, it can be left blank.

## Output Format
Return your analysis in the following JSON format:

```json
{
    "new_entities": [
        {
            "name": "object_name",
            "info": "description of the object"
        }
    ],
    "new_relationships": [
        {
            "relation_name": "relationship_type",
            "source_name": "source_entity_name", // no person
            "target_name": "target_entity_name", // no person
            "relation_info": "description of the relationship",
            "start_time": "hh:mm:ss",
            "end_time": "hh:mm:ss" // optional
        }
    ],
    "new_actions": [
        {
            "action_name": "action_verb",
            "action_info": "description of the action",
            "time_range": "{period}",
            "agent": "person_name",
            "patient": "affected_entity_name",  // optional
            "instrument": "tool_entity_name",   // optional
            "source": "starting_location",      // optional
            "target": "ending_location"         // optional
            "prev_action_id": "previous_action_id" // required, make sure the sequence of actions is correct. format: {agent}_{action_name}_{patient}_{time_range}
        }
    ]
}
```
Notes:
- Only include new information not already present in the current scene graph
- Use underscores (_) instead of spaces in relationship and action names
- Make relationship and action descriptions concise but informative
- All entity names must match existing names in the scene graph or be added as new entities
- Don't add relationships or actions like "not xx"
""".strip()

        prompt_ablation = """
        # Related Question
        {question}

        # Scene Graph Update Request

        ## Current Scene Graph Information
        {subgraph}

        ## Input Text
        The following text describes a new Q&A with VLM within the period '{period}':
        Q: {llm_instruction}
        A: {input_text}

        ## Task
        Based on the input text, extract new objects, relationships, and actions that should be added to the current scene graph for this period.
        
        ## Output Format
        Return your analysis in the following JSON format:

        ```json
        {
            "new_entities": [
                {
                    "name": "object_name",
                    "info": "description of the object"
                }
            ],
            "new_relationships": [
                {
                    "relation_name": "relationship_type",
                    "source_name": "source_entity_name", // no person
                    "target_name": "target_entity_name", // no person
                    "relation_info": "description of the relationship",
                    "start_time": "hh:mm:ss",
                    "end_time": "hh:mm:ss" // optional
                }
            ],
            "new_actions": [
                {
                    "action_name": "action_verb",
                    "action_info": "description of the action",
                    "time_range": "{period}",
                    "agent": "person_name",
                    "patient": "affected_entity_name",  // optional
                    "instrument": "tool_entity_name",   // optional
                    "source": "starting_location",      // optional
                    "target": "ending_location"         // optional
                    "prev_action_id": "previous_action_id" // required, make sure the sequence of actions is correct. format: {agent}_{action_name}_{patient}_{time_range}
                }
            ]
        }
        ```
        """.strip()

        prompt = prompt.replace("{input_text}", input_text)
        prompt = prompt.replace("{subgraph}", subgraph_json_text)
        prompt = prompt.replace("{period}", period)
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{llm_instruction}", llm_instruction)
        # Call LLM.
        update_data = None
        msgs = [{"role": "user", "content": prompt}]
        response = None
        for i in range(5):
            try:
                response = basic_llm_chat(msgs, temperature=0)
                update_data = utils.parse_json_text(response)
                break
            except Exception as e:
                print(e)
                continue

        # Parse the JSON output and add entities/relationships/actions.
        # Add new entities.
        if update_data is None:
            print("Failed to update the relation graph")
            return False
        if "new_entities" in update_data and update_data["new_entities"]:
            for entity in update_data["new_entities"]:
                entity_type = entity.get("type", "").lower()
                entity_name = entity.get("name", "")
                entity_info = entity.get("info", "")
                # Replace spaces with underscores.
                entity_name = re.sub(r'\s+', '_', entity_name)

                if not entity_name:
                    continue

                self.relation_graph.add_update_objects(entity_name, period, entity_info)
                # Update navigation graph.
                self.navigation_graph.add_objs([entity_name], period)
                print("Add new entity:", entity_name, entity_info)


        # Add new relationships.
        if "new_relationships" in update_data and update_data["new_relationships"]:
            for relation in update_data["new_relationships"]:
                relation_name = relation.get("relation_name", "")
                source_name = relation.get("source_name", "")
                target_name = relation.get("target_name", "")
                relation_info = relation.get("relation_info", "")
                start_time = relation.get("start_time", "")
                end_time = relation.get("end_time", None)

                if not relation_name or not source_name or not target_name or not start_time:
                    continue

                # Replace spaces with underscores.
                source_name = re.sub(r'\s+', '_', source_name)
                target_name = re.sub(r'\s+', '_', target_name)


                self.relation_graph.add_relation(
                    source_name,
                    target_name,
                    relation_name,
                    relation_info,
                    start_time,
                    end_time
                )
                print("Add new relationship:", relation_name, source_name, target_name, start_time, end_time)

        # Add new actions.
        if "new_actions" in update_data and update_data["new_actions"]:
            for action in update_data["new_actions"]:

                action_name = action.get("action_name", "")
                action_info = action.get("action_info", "")
                time_range = action.get("time_range", period)  # default to period
                agent = action.get("agent", "")
                patient = action.get("patient", None)
                instrument = action.get("instrument", None)
                source = action.get("source", None)
                target = action.get("target", None)
                prev_action_id = action.get("prev_action_id", None)

                if not action_name or not agent:
                    continue

                # Replace spaces with underscores.
                action_name = re.sub(r'\s+', '_', action_name)
                agent = re.sub(r'\s+', '_', agent)
                patient = re.sub(r'\s+', '_', patient) if patient else None
                instrument = re.sub(r'\s+', '_', instrument) if instrument else None
                source = re.sub(r'\s+', '_', source) if source else None
                target = re.sub(r'\s+', '_', target) if target else None

                # Extract start/end from the time_range.
                if "-" in time_range:
                    start_time, end_time = time_range.split("-")
                else:
                    start_time, end_time = time_range, None

                # Compose the full time range string.
                full_time_range = f"{start_time}-{end_time}" if end_time else start_time

                new_action_id = self.relation_graph.add_action(
                    action_name,
                    action_info,
                    full_time_range,
                    agent,
                    patient,
                    instrument,
                    source,
                    target,
                    prev_action_id
                )
                # Update action sequence in navigation graph.
                self.navigation_graph.periods_to_activities[period] = self.relation_graph.get_all_actions_in_time_range(period)
                # Log the newly added action.
                print(f"Add new action: {action_name}, {action_info}, {agent}, {full_time_range}, {patient}, {instrument}, {source}, {target}, {prev_action_id}")

            # Call LLM to add new key entities.
        if response:
            msgs.append({"role": "assistant", "content": response})
            prompt_2 = """
Related Question:
{question}

Input text:
{input_text}

Current key entities:
{key_entities}

All entity names in this period:
{all_entities}

Task:
Based on the update of the scene diagram, please analyze the problem and the input text to see if there are any new entities related to the question that need to be added as key entities. If so, output the corresponding entity name.

Output format:
```json
{
    "new_key_entities": ["entity_name"], // can be empty
}
```
            """.strip()
            prompt_2 = prompt_2.replace("{question}", question).replace("{input_text}", input_text).replace("{key_entities}", str(self.marked_entity_names)).replace("{all_entities}", str(self.navigation_graph.periods_to_obj_names[period]))
            msgs.append({"role": "user", "content": prompt_2})
            # Call LLM.
            for i in range(5):
                try:
                    response = basic_llm_chat(msgs, temperature=0)
                    response_dict = utils.parse_json_text(response)
                    if "new_key_entities" in response_dict:
                        new_key_entities = response_dict["new_key_entities"]
                        if new_key_entities and len(new_key_entities) > 0:
                            # Add new key entities.
                            self.add_marked_entities(new_key_entities)
                            print("Add new key entity:", new_key_entities)
                    break
                except Exception as e:
                    print(e)
                    continue


        return True


    def init_persons_and_areas(self, input_text):
        """
        Initialize persons and areas from the VLM output.
        """
        print("Initialize persons and areas")
        st_time = time.time()
        prompt = """# The following is a segmented description of a POV video:
{input_text}

# Task:
Based on the video description, the main people contained in the video are extracted, as well as the scene areas involved.
An area refers to a room or space in a scene, such as "bedroom", "workshop", "playground", etc.
You should also determine the time range in which each area appears in the video based on the description.
The output format is as follows:
```json
{
    "persons": [
        {
            "name": "person_name",
            "info": "description of the person"
        }
    ],
    "areas": [
        {
            "name": "area_name",
            "time_range": "hh:mm:ss-hh:mm:ss",
            "info": "brief description of the area"
        }
    ]
}
```
"""
        prompt_ablation = """# The following is a segmented description of a POV video:
        {input_text}

        # Task:
        Based on the video description, the main people contained in the video are extracted, as well as the scene areas involved.
        The output format is as follows:
        ```json
        {
            "persons": [
                {
                    "name": "person_name",
                    "info": "concise description of the person"
                }
            ],
            "areas": [
                {
                    "name": "area_name",
                    "time_range": "hh:mm:ss-hh:mm:ss",
                    "info": "brief description of the area"
                }
            ]
        }
        ```
        """
        prompt = prompt.replace("{input_text}", input_text)
        msgs = [{"role": "user", "content": prompt}]
        # Call LLM to output JSON and parse it (retry up to 5 times).
        for i in range(5):
            try:
                response = basic_llm_chat(msgs)
                print(response)
                response = utils.parse_json_text(response)
                # Validate format.
                if "persons" in response and "areas" in response:
                    persons = response["persons"]
                    areas = response["areas"]
                    # Add persons and areas.
                    self.navigation_graph.add_persons(persons)
                    self.navigation_graph.add_areas(areas)
                    # Update relation graph.
                    for person in persons:
                        self.relation_graph.add_person(person["name"], person["info"])
                    for area in areas:
                        self.relation_graph.add_area(area["name"], area["time_range"], area["info"])
                    ed_time = time.time()
                    print(f"time used: {ed_time - st_time:.2f}s")
                    return
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                continue

    def init_obj(self, periods_description, question, video_path):
        """
        Initialize objects (currently only YOLO-detected objects are initialized).
        """
        print("init obj")
        st_time = time.time()
        llm_parse_obj_list = """
        Video overall description: 
        {video_description}
        Question: {question}
        Please extract all the objects based on the video overall description and the question, list them while excluding floor, and wall.
        - The object name can have a color or material modifier. 
        - There should be no ambiguous object names. Don't use "or". Object names must be singular.
        - The objects included in the question must be listed, even though they may not appear in the video description.
        - The entire video is continuous. However, since each time period is described independently, it is possible that the same object may have different names or descriptions in different time periods. You need to identify this situation and use only one name to represent this object.
        - All object information must be consistent with the above description. Do not make up your own information.
        Output in JSON format:  

        ```json
        {
            "objects": [
                {
                    "name": "object_name",
                    "info": "optional, concise info about the object",
                    "time_periods": ["hh:mm:ss-hh:mm:ss", ...] // time periods when the object appears
                }
            ],
        }
        ```
        """

        llm_parse_obj_list_ablation = """
                Video overall description: 
                {video_description}
                Question: {question}
                Please extract all the objects based on the video overall description and the question, list them while excluding floor, and wall.
                Output in JSON format:  

                ```json
                {
                    "objects": [
                        {
                            "name": "object_name",
                            "info": "optional, info about the object",
                            "time_periods": ["hh:mm:ss-hh:mm:ss", ...] // time periods when the object appears
                        }
                    ],
                }
                ```
                """

        for i in range(3):
            try:
                # Let the LLM reorganize the object list.
                response_obj_list = basic_llm_chat([{"role": "user",
                                                          "content": llm_parse_obj_list.replace("{video_description}",
                                                                                                periods_description).replace(
                                                              "{question}", question)}])
                print(response_obj_list)
                # Parse JSON.
                response_obj_list_json = utils.parse_json_text(response_obj_list)
                if "objects" in response_obj_list_json:
                    # Validate format.
                    objects = response_obj_list_json["objects"]
                    # print(objects)
                    # Normalize to an object name list.
                    obj_names = []
                    obj_names_to_info = {}
                    for obj in objects:
                        if "name" in obj:
                            obj_names.append(obj["name"])
                            if "info" in obj:
                                obj_names_to_info[obj["name"]] = obj["info"]
                                # Add to relation graph.
                                self.relation_graph.add_update_objects(obj["name"], "", obj["info"])
                            if "time_periods" in obj:
                                # Add to navigation graph.
                                for period in obj["time_periods"]:
                                    # A period may overlap multiple nav periods; add the object to all overlaps.
                                    time_periods = spilit_video.find_time_range(period, self.navigation_graph.periods_infos.keys())
                                    if time_periods:
                                        for time_period in time_periods:
                                            self.navigation_graph.add_objs([obj["name"]], time_period)

                        else:
                            continue

                    # filtered_objects = object_detection.track_and_filter_objects(video_path, classes=obj_names,
                    #                                                              conf_threshold=0.1, iou_threshold=0.7,
                    #                                                              min_duration_seconds=0.5,
                    #                                                              min_avg_conf=0.2,
                    #                                                              vid_stride=6)
                    filtered_objects = {}

                    # Collect detected object classes.
                    detected_objects = set()
                    for obj_id, obj_value in filtered_objects.items():
                        detected_objects.add(obj_value["class"])
                        obj_info = obj_names_to_info.get(obj_value["class"], None)
                        # Convert seconds to hh:mm:ss.
                        start_time = obj_value["first_second"]
                        end_time = obj_value["last_second"]
                        start_time_hhmmss = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"
                        end_time_hhmmss = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{int(end_time % 60):02d}"
                        time_range = f"{start_time_hhmmss}-{end_time_hhmmss}"
                        # Add object to relation graph.
                        self.relation_graph.add_update_objects(obj_value["class"], "", obj_info)
                    print(detected_objects)
                    # Update navigation graph.
                    self.add_detected_objects(filtered_objects)
                    ed_time = time.time()
                    print(f"time used: {ed_time - st_time:.2f}s")

                    break
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                response_obj_list = None
                continue

    def add_detected_objects(self, filtered_objects):
        """
        Add YOLO-detected objects into the navigation graph.
        :param filtered_objects: Filtered object dict returned by track_and_filter_objects
        """
        # Convert period strings to seconds.
        period_times = {}
        for period in self.navigation_graph.periods_infos.keys():
            start_time_str, end_time_str = period.split("-")
            start_time = utils.time_to_seconds(start_time_str)
            end_time = utils.time_to_seconds(end_time_str)
            period_times[period] = (start_time, end_time)

        # Iterate over detected objects.
        for obj_id, obj_info in filtered_objects.items():
            obj_class = obj_info["class"]
            first_second = obj_info["first_second"]
            last_second = obj_info["last_second"]

            # Find which period(s) the object should belong to.
            for period, (start_time, end_time) in period_times.items():
                # If the object's appearance overlaps the period.
                if (first_second <= end_time or last_second >= start_time):
                    # Add object to navigation graph.
                    self.navigation_graph.add_objs([obj_class], period)

    def init_obj_rel(self, periods_description):
        """
        Initialize non-action relationships.
        """
        print("init obj rel")
        st_time = time.time()
        prompt = """
# Video Segments Description  
{description}  

# Task Definition  
The above text describes a POV video segment. Based on the video description provided, extract all non-action relations related to the entities in the list.  
An action refers to a simple movement performed by a person in the video, such as "picking up," "stirring," "opening," etc. Actions are executed by a person and applied to an entity.  
Relations differ from these dynamic actions—they represent static states, such as "located at," "near," "above," "contains," or "connected to."  
A relation exists between two entities and is represented as a triple with direction: source entity → relation → target entity.  
Relations also have attributes. In addition to the relation name, you may include descriptive information to provide further details. Furthermore, relations have start and end time attributes in the format "hh:mm:ss." The start time is mandatory and indicates when the relation first appears, while the end time is optional and marks when the relation ceases. The appearance and termination of relations are typically caused by interactive actions. 
Spaces are not allowed in relation names. If the name consists of multiple words, use "_" to connect them.
When creating a relation, the source and target entities must come from the following entity list:  

## Entity List  
Characters: {chars}
Areas: {areas}
Objects: {objs}

## Output Format  
```json  
{  
    "relations": [  
        {  
            "relation_name": "",  
            "source_name": "",  
            "target_name": "",  
            "relation_info": "optional, additional info",  
            "start_time": "hh:mm:ss",  
            "end_time": "hh:mm:ss"    // optional  
        },  
    ]  
}  
```
        """.strip()

        prompt_ablation = """
        # Video Segments Description  
        {description}  

        # Task Definition  
        The above text describes a POV video segment. Based on the video description provided, extract all non-action relations related to the entities in the list.  
        Spaces are not allowed in relation names. If the name consists of multiple words, use "_" to connect them.
        When creating a relation, the source and target entities must come from the following entity list:  

        ## Entity List  
        Characters: {chars}
        Areas: {areas}
        Objects: {objs}

        ## Output Format  
        ```json  
        {  
            "relations": [  
                {  
                    "relation_name": "",  
                    "source_name": "",  
                    "target_name": "",  
                    "relation_info": "",  
                    "start_time": "hh:mm:ss",  
                    "end_time": "hh:mm:ss"    // optional  
                },  
            ]  
        }  
        ```
                """.strip()

        prompt = prompt.replace("{description}", periods_description)
        prompt = prompt.replace("{chars}", self.get_all_char_infos())
        prompt = prompt.replace("{areas}", str(self.navigation_graph.area_names))
        prompt = prompt.replace("{objs}", str(self.navigation_graph.obj_names))
        msgs = [{"role": "user", "content": prompt}]
        # Call LLM to output JSON and parse it (retry up to 5 times).
        for i in range(5):
            try:
                response = basic_llm_chat(msgs)
                print(response)
                response = utils.parse_json_text(response)
                # Validate format.
                if "relations" in response:
                    relations = response["relations"]
                    # Optionally validate entity names exist in the navigation graph.
                    for relation in relations:
                        if "source_name" in relation and "target_name" in relation:
                            source_name = relation["source_name"]
                            target_name = relation["target_name"]
                            # # Validate source/target entity names exist in the navigation graph.
                            # if source_name not in self.navigation_graph.person_names and \
                            #         source_name not in self.navigation_graph.area_names and \
                            #         source_name not in self.navigation_graph.obj_names:
                            #     print(f"Source entity '{source_name}' is not in the navigation graph")
                            #     continue
                            # if target_name not in self.navigation_graph.person_names and \
                            #         target_name not in self.navigation_graph.area_names and \
                            #         target_name not in self.navigation_graph.obj_names:
                            #     print(f"Target entity '{target_name}' is not in the navigation graph")
                            #     continue
                            # Add relation to relation graph.
                            self.relation_graph.add_relation(source_name, target_name, relation["relation_name"], relation["relation_info"], relation["start_time"], relation.get("end_time", None))
                    ed_time = time.time()
                    print(f"time used: {ed_time - st_time:.2f}s")
                    break
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                continue


    def add_marked_entities(self, entity_names):
        """
        Add marked (key) entities.
        """
        self.marked_entity_names.update(entity_names)

    def get_all_char_infos(self):
        # Build person info text.
        char_infos = ""
        for char in self.navigation_graph.person_names:
            char_infos += f"{char}: {self.navigation_graph.get_person_info(char)}, "
        return char_infos

    def init_action(self, periods_description):
        """
        Initialize actions.
        """
        print("init action")
        st_time = time.time()
        prompt = """
# Video Segments Description  
{description}  

# Task Definition  
The above text describes a first-person video segment. Based on the video description and the list of detected entities, extract the actions contained in the video.  

## Action:  
An action refers to a simple movement performed by a person in the video, such as "picking up," "stirring," "opening," etc. Actions are executed by a person and applied to an entity. An action may involve the following entities: **Agent**, **Patient**, **Instrument**, **Source**, **Target**.  
- **Agent**: The initiator of the action, usually a person (required).  
- **Patient**: The entity affected by the action.  
- **Instrument**: The tool used to perform the action.  
- **Source**: The initial location of the affected entity before the action.  
- **Target**: The location of the affected entity after the action is completed.  
Only the **Agent** is mandatory; all other entities are optional. Source and Target are usually used in actions involving movement, and Instrument is used in actions involving tools.
Each of the above items can only contain one entity.

An action includes the following attributes: action name, action description, and the time range of the action.  
The output list of actions must follow the chronological order of the original description.  

### Examples:  
1. Description: A man is doing yoga.  
   ```json  
   {"action_name": "doing yoga", "agent": "man", "time_range": "00:00:00-00:00:30"}  
   ```  
2. Description: A takes an apple out of the brown bag and places it on the table.  
   ```json  
   {"action_name": "take out and place", "agent": "A", "patient": "apple", "source": "brown bag", "target": "table", "time_range": "00:00:30-00:01:00"}  
   ```  
3. Description: C cuts a potato with a knife.  
   ```json  
   {"action_name": "cut", "agent": "C", "patient": "potato", "instrument": "knife", "time_range": "00:01:00-00:01:30"}  
   ```  

## Entity List  
Make sure the entity name you fill in is exactly the same as that in the following list:
Characters: {chars}  
Areas: {areas}  
Objects: {objs}  

## Output Format  
```json  
{  
    "actions": [  // Must follow chronological order  
        {  
            "action_name": "",  
            "time_range": "hh:mm:ss-hh:mm:ss",  // Must be consistent with the start and end times in the segment description
            "agent": "",    // Required  
            "patient": "",  
            "instrument": "",  
            "source": "",  
            "target": ""  
        },  
    ]  
}  
```
        """.strip()

        prompt_ablation = """
        # Video Segments Description  
        {description}  

        # Task Definition  
        The above text describes a first-person video segment. Based on the video description and the list of detected entities, extract the actions contained in the video.  

        ## Action:  
        An action refers to a simple movement performed by a person in the video, such as "picking up," "stirring," "opening," etc. Actions are executed by a person and applied to an entity. An action may involve the following entities: **Agent**, **Patient**, **Instrument**, **Source**, **Target**.  
        - **Agent**: The initiator of the action, usually a person (required).  
        - **Patient**: The entity affected by the action.  
        - **Instrument**: The tool used to perform the action.  
        - **Source**: The initial location of the affected entity before the action.  
        - **Target**: The location of the affected entity after the action is completed.  
        Only the **Agent** is mandatory; all other entities are optional. Source and Target are usually used in actions involving movement, and Instrument is used in actions involving tools.
        Each of the above items can only contain one entity.

        An action includes the following attributes: action name, action description, and the time range of the action.  
        The output list of actions must follow the chronological order of the original description.  

        ### Examples:  
        1. Description: A man is doing yoga.  
           ```json  
           {"action_name": "doing yoga", "agent": "man", "action_info": "The man is doing yoga.", "time_range": "00:00:00-00:00:30"}  
           ```  
        2. Description: A takes an apple out of the brown bag and places it on the table.  
           ```json  
           {"action_name": "take out and place", "agent": "A", "patient": "apple", "source": "brown bag", "target": "table", "action_info": "A takes the apple out of the brown bag and places it on the table.", "time_range": "00:00:30-00:01:00"}  
           ```  
        3. Description: C cuts a potato with a knife.  
           ```json  
           {"action_name": "cut", "agent": "C", "patient": "potato", "instrument": "knife", "action_info": "C cuts the potato with a knife.", "time_range": "00:01:00-00:01:30"}  
           ```  

        ## Entity List  
        Make sure the entity name you fill in is exactly the same as that in the following list:
        Characters: {chars}  
        Areas: {areas}  
        Objects: {objs}  

        ## Output Format  
        ```json  
        {  
            "actions": [  // Must follow chronological order  
                {  
                    "action_name": "",  
                    "action_info": "",  
                    "time_range": "hh:mm:ss-hh:mm:ss",  // Must be consistent with the start and end times in the segment description
                    "agent": "",    // Required  
                    "patient": "",  
                    "instrument": "",  
                    "source": "",  
                    "target": ""  
                },  
            ]  
        }  
        ```
                """.strip()

        prompt = prompt.replace("{description}", periods_description)
        prompt = prompt.replace("{chars}", self.get_all_char_infos())
        prompt = prompt.replace("{areas}", str(self.navigation_graph.area_names))
        prompt = prompt.replace("{objs}", str(self.navigation_graph.obj_names))
        msgs = [{"role": "user", "content": prompt}]
        # Call LLM to output JSON and parse it (retry up to 5 times).
        for i in range(5):
            try:
                response = basic_llm_chat(msgs)
                print(response)
                response = utils.parse_json_text(response)
                # Validate format.
                if "actions" in response:
                    actions = response["actions"]
                    # Optionally validate entity names exist in the navigation graph.
                    prev_action_id = None
                    for action in actions:
                        if "agent" in action:
                            agent_name = action["agent"]
                            # Validate agent exists in the navigation graph.
                            # if agent_name not in self.navigation_graph.person_names:
                            #     raise ValueError(f"Entity '{agent_name}' is not in the navigation graph")
                            prev_action_id = self.relation_graph.add_action(action["action_name"], action.get("action_info", None), action["time_range"], agent_name, action.get("patient", None), action.get("instrument", None), action.get("source", None), action.get("target", None), prev_action_id)
                            self.navigation_graph.add_activity(action["action_name"], remove_time_decimals(action["time_range"]))
                    ed_time = time.time()
                    print(f"time used: {ed_time - st_time:.2f}s")
                    break
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                continue


    def get_vid_period_segment_path(self, period):
        """
        Get the file path for a video segment of the given period.
        """
        # Check whether the period exists.
        if period not in self.navigation_graph.video_segments_to_files:
            raise ValueError(f"Period '{period}' does not exist")
        return self.navigation_graph.video_segments_to_files[period]


    def output_clue_subgraph(self, entity_names, max_path_length=5):
        """
        Output a subgraph for the given key entity names.

        :param entity_names: List of key entity names
        :param max_path_length: Maximum path length when searching paths between nodes
        :return: Text summary of the subgraph
        """
        return self.relation_graph.format_subgraph(self.relation_graph.extract_subgraph_by_nodes(entity_names, max_path_length=max_path_length))

