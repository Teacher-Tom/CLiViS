"""Neo4j-backed relation graph."""

from clivis.graph.common import *

class RelationGraph(object):
    def __init__(self):
        self.database = "neo4j"
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        self.driver.verify_connectivity()
        self.clear_database()
        self._setup_constraints_and_indexes()
        # Configure Neo4j driver logging (suppress warnings).
        import logging
        logging.getLogger("neo4j").setLevel(logging.ERROR)

    def close(self):
        """Close the Neo4j driver when no longer needed."""
        if self.driver:
            self.driver.close()

    def _setup_constraints_and_indexes(self):
        """Create necessary constraints and indexes."""
        with self.driver.session(database=self.database) as session:
            # Create an index on `name` for each node label.
            for label in [label.value for label in NodeLabels]:
                session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)")


    def get_node_info(self, node_name, node_label=None):
        """
        Read a node's properties.

        :param node_name: Node name
        :param node_label: Optional node label; if None, match by name only
        """
        with self.driver.session() as session:
            if node_label is None:
                query = f"MATCH (n) WHERE n.name = '{node_name}' RETURN n"
            else:
                query = f"MATCH (n:{node_label}) WHERE n.name = '{node_name}' RETURN n"
            result = session.run(query)
            nodes = [record[0] for record in result]
            if len(nodes) == 0:
                return None
            else:
                node_info = dict(nodes[0])
                return node_info


    def add_update_node(self, node_name, node_label, attr_dict=None):
        """
        Add a node; if it already exists, update its properties.

        :param node_name: Node name
        :param node_label: Node label
        :param attr_dict: Node property dict
        :return: True if node was added/updated successfully
        """
        if attr_dict is None:
            attr_dict = {}

        # Ensure the node name is included.
        attr_dict['name'] = node_name

        # Escape single quotes in string properties.
        for k,v in attr_dict.items():
            attr_dict[k] = v.replace("'", "\\'")

        # Convert property dict to Cypher-friendly string.
        props_str = ', '.join([f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}" for k, v in attr_dict.items()])

        with self.driver.session(database=self.database) as session:
            try:
                # Use MERGE to avoid duplicates: update if exists, create otherwise.
                query = f"MERGE (n:{node_label} {{name: '{node_name}'}}) SET n = {{{props_str}}} RETURN n"
                result = session.run(query)

                # Check whether execution succeeded.
                records = list(result)
                return len(records) > 0
            except Exception as e:
                print(f"Error adding/updating node: {e}")
                return False

    def add_person(self, person_name, info):
        """
        Add a person node.

        :param person_name: Person name
        :param info: Person info text
        :return: Whether the node was added successfully
        """
        # Extract properties.
        attr_dict = {
            "name": person_name,
            "info": info
        }

        # Use the generic node upsert helper.
        return self.add_update_node(person_name, NodeLabels.PERSON.value, attr_dict)

    def add_update_objects(self, obj_name, time_range, obj_info):
        """
        Add/update an object node.

        :param obj_name: Object name
        :param time_range: Time range, hh:mm:ss-hh:mm:ss
        :param obj_info: Object info text
        :return: Whether the node was added successfully
        """
        # Build property dict.
        attr_dict = {
            "name": obj_name,
            "info": obj_info,
            "time_range": time_range
        }

        # Add/update object node.
        success = self.add_update_node(obj_name, NodeLabels.OBJECT.value, attr_dict)

        return success

    def add_area(self, area_name, time_range, area_info):
        """
        Add an area node.

        :param area_name: Area name
        :param time_range: Time range, hh:mm:ss-hh:mm:ss
        :param area_info: Area info text
        :return: Whether the node was added successfully
        """
        # Build property dict.
        attr_dict = {
            "name": area_name,
            "info": area_info,
            "time_range": time_range
        }

        # Add/update area node.
        success = self.add_update_node(area_name, NodeLabels.AREA.value, attr_dict)

        return success

    def get_relation_info(self, relation_name):
        """
        Read a relation type's properties (first match).

        :param relation_name: Relation type name
        """
        with self.driver.session() as session:
            query = f"MATCH ()-[r]->() WHERE type(r) = '{relation_name}' RETURN r"
            result = session.run(query)
            relations = [record[0] for record in result]
            if len(relations) == 0:
                return None
            else:
                relation_info = dict(relations[0])
                return relation_info

    def get_relations_of_node(self, node_name, node_label=None):
        """
        Get all relations connected to a node (both directions), including the other endpoint.

        :param node_name: Node name
        :param node_label: Optional node label
        :return: Dict containing outgoing and incoming relations in the form:
            {'outgoing': [{'relation_type': str, 'target_node': dict, 'relation_props': dict}],
             'incoming': [{'relation_type': str, 'source_node': dict, 'relation_props': dict}]}
        """
        relations = {
            'outgoing': [],
            'incoming': []
        }

        with self.driver.session(database=self.database) as session:
            try:
                # Build query condition.
                label_condition = f":{node_label}" if node_label else ""

                # Outgoing relations.
                outgoing_query = f"""
                MATCH (n{label_condition} {{name: $node_name}})-[r]->(target)
                RETURN type(r) AS relation_type, target, r
                """
                outgoing_result = session.run(outgoing_query, node_name=node_name)

                # Incoming relations.
                incoming_query = f"""
                MATCH (source)-[r]->(n{label_condition} {{name: $node_name}})
                RETURN type(r) AS relation_type, source, r
                """
                incoming_result = session.run(incoming_query, node_name=node_name)

                # Process outgoing results.
                for record in outgoing_result:
                    relations['outgoing'].append({
                        'relation_type': record['relation_type'],
                        'target_node': dict(record['target']),
                        'relation_props': dict(record['r'])
                    })

                # Process incoming results.
                for record in incoming_result:
                    relations['incoming'].append({
                        'relation_type': record['relation_type'],
                        'source_node': dict(record['source']),
                        'relation_props': dict(record['r'])
                    })

                return relations

            except Exception as e:
                print(f"Error getting node relations: {e}")
                return relations

    def get_paths_between_nodes(self, node_a_name, node_b_name, max_step=10, dual_direction=True):
        """
        Get all possible paths between two nodes without repeating edges.

        :param node_a_name: Node A name
        :param node_b_name: Node B name
        :param max_step: Maximum hop length
        :param dual_direction: If True, also search paths from B to A
        :return: List of paths; each path is a list of dict items describing nodes and relations
        """
        # Ensure nodes are different.
        if node_a_name == node_b_name:
            print(f"Start and end nodes are the same: {node_a_name}")
            return []

        with self.driver.session(database=self.database) as session:
            try:
                # Base query for A -> B.
                query = """
                MATCH (a {name: $node_a_name}), (b {name: $node_b_name})
                """

                if dual_direction:
                    # Dual-direction search: include A->B and B->A paths.
                    query += """
                    CALL {
                        WITH a, b
                        MATCH p = (a)-[*1..%d]->(b)
                        WHERE all(rel IN relationships(p) WHERE
                              size([r IN relationships(p) WHERE type(r) = type(rel)
                                    AND startNode(r) = startNode(rel)
                                    AND endNode(r) = endNode(rel)]) = 1)
                        RETURN p

                        UNION

                        WITH a, b
                        MATCH p = (b)-[*1..%d]->(a)
                        WHERE all(rel IN relationships(p) WHERE
                              size([r IN relationships(p) WHERE type(r) = type(rel)
                                    AND startNode(r) = startNode(rel)
                                    AND endNode(r) = endNode(rel)]) = 1)
                        RETURN p
                    }
                    """ % (max_step, max_step)
                else:
                    # One-way search: only A->B paths.
                    query += """
                    CALL {
                        WITH a, b
                        MATCH p = (a)-[*1..%d]->(b)
                        WHERE all(rel IN relationships(p) WHERE
                              size([r IN relationships(p) WHERE type(r) = type(rel)
                                    AND startNode(r) = startNode(rel)
                                    AND endNode(r) = endNode(rel)]) = 1)
                        RETURN p
                    }
                    """ % max_step

                # Finalize query.
                query += """
                RETURN p AS path
                LIMIT 100
                """

                result = session.run(query, node_a_name=node_a_name, node_b_name=node_b_name)

                paths = []
                for record in result:
                    path = record['path']
                    path_segments = []

                    # Process nodes and relationships.
                    nodes = path.nodes
                    relationships = path.relationships

                    # First node.
                    path_segments.append({
                        'type': 'node',
                        'label': list(nodes[0].labels)[0] if nodes[0].labels else None,
                        'name': nodes[0]['name'],
                        'properties': dict(nodes[0]),
                        'is_start': nodes[0]['name'] == node_a_name
                    })

                    # Iterate relationships and subsequent nodes.
                    for i, rel in enumerate(relationships):
                        # Relationship.
                        path_segments.append({
                            'type': 'relationship',
                            'relation_type': rel.type,
                            'properties': dict(rel),
                            'direction': 'outgoing' if nodes[i] == rel.start_node else 'incoming'
                        })

                        # Next node.
                        next_node = nodes[i + 1]
                        path_segments.append({
                            'type': 'node',
                            'label': list(next_node.labels)[0] if next_node.labels else None,
                            'name': next_node['name'],
                            'properties': dict(next_node),
                            'is_start': next_node['name'] == node_a_name
                        })

                    paths.append(path_segments)

                return paths
            except Exception as e:
                print(f"Error getting paths between nodes: {e}")
                return []

    def add_relation(self, node_a_name, node_b_name, relation_type, relation_info, start_time, end_time=None):
        """
        Add a non-action relationship.

        :param node_a_name: Source node name
        :param node_b_name: Target node name
        :param relation_type: Relationship type
        :param relation_info: Relationship description
        :param start_time: Relationship start time, hh:mm:ss
        :param end_time: Relationship end time (optional)
        :return: Whether the relationship was added successfully
        """

        with self.driver.session(database=self.database) as session:
            try:
                # Build relationship properties.
                props = {
                    "info": relation_info,
                    "start_time": start_time
                }

                # Add optional end_time.
                if end_time:
                    props["end_time"] = end_time

                # Convert property dict to a Cypher-friendly string.
                # props_str = ', '.join(
                #     [f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}" for k, v in props.items()])

                # Escape single quotes in properties.
                props_str_items = []
                for k, v in props.items():
                    if isinstance(v, str):
                        escaped_v = v.replace("'", "\\'")
                        props_str_items.append(f"{k}: '{escaped_v}'")
                    else:
                        props_str_items.append(f"{k}: {v}")
                props_str = ', '.join(props_str_items)


                # Build Cypher query. Use MERGE to avoid duplicate relationships.
                # First match the two nodes, then create/merge the relationship between them.
                query = f"""
                MATCH (a {{name: $node_a_name}}), (b {{name: $node_b_name}})
                MERGE (a)-[r:{relation_type} {{{props_str}}}]->(b)
                RETURN r
                """

                result = session.run(query, node_a_name=node_a_name, node_b_name=node_b_name)

                # Check whether the relationship was added successfully.
                return len(list(result)) > 0

            except Exception as e:
                print(f"Error adding relationship: {e}")
                return False

    def add_action(self, action_name, action_info, time_range, node_agent_name, node_patient_name=None,
                   node_instrument_name=None, node_source_name=None, node_target_name=None,
                   prev_action_id=None):
        """
        Add an action. The action itself is stored as an Activity node and linked to other entities.
        Supports a linked-list structure to record action ordering, possibly across periods.

        :param action_name: Action name
        :param action_info: Action description
        :param time_range: Time range, hh:mm:ss-hh:mm:ss
        :param node_agent_name: Agent entity name (required)
        :param node_patient_name: Patient entity name (optional)
        :param node_instrument_name: Instrument entity name (optional)
        :param node_source_name: Source entity name (optional)
        :param node_target_name: Target entity name (optional)
        :param prev_action_id: Previous action id (optional)
        :return: Action id on success; False on failure
        """
        with self.driver.session(database=self.database) as session:
            try:
                # Escape single quotes in all string parameters.
                action_name = action_name.replace("'", "\\'") if action_name else None
                action_info = action_info.replace("'", "\\'") if action_info else None
                time_range = time_range.replace("'", "\\'") if time_range else None
                node_agent_name = node_agent_name.replace("'", "\\'") if node_agent_name else None
                node_patient_name = node_patient_name.replace("'", "\\'") if node_patient_name else None
                node_instrument_name = node_instrument_name.replace("'", "\\'") if node_instrument_name else None
                node_source_name = node_source_name.replace("'", "\\'") if node_source_name else None
                node_target_name = node_target_name.replace("'", "\\'") if node_target_name else None
                prev_action_id = prev_action_id.replace("'", "\\'") if prev_action_id else None

                action_id = self.compose_action_id(action_name, time_range, node_agent_name, node_patient_name)
                action_id = action_id.replace("'", "\\'")  # ensure generated id is escaped

                # Build query to create the action node.
                query = f"""
                CREATE (action:Activity {{name: '{action_name}', info: '{action_info}', time_range: '{time_range}', action_id: '{action_id}', agent_name: '{node_agent_name}'"""

                if node_patient_name:
                    query += f", patient_name: '{node_patient_name}'"
                if node_instrument_name:
                    query += f", instrument_name: '{node_instrument_name}'"
                if node_source_name:
                    query += f", source_name: '{node_source_name}'"
                if node_target_name:
                    query += f", target_name: '{node_target_name}'"
                if prev_action_id:
                    query += f", prev_action_id: '{prev_action_id}'"
                query += "})"

                # Build relationship queries.
                query += f"\nWITH action MATCH (agent:{NodeLabels.PERSON.value} {{name: '{node_agent_name}'}})"
                query += "\nCREATE (agent)-[:PERFORMS]->(action)"

                if node_patient_name:
                    query += f"\nWITH action MATCH (patient {{name: '{node_patient_name}'}})"
                    query += "\nCREATE (action)-[:AFFECTS]->(patient)"
                if node_instrument_name:
                    query += f"\nWITH action MATCH (instrument {{name: '{node_instrument_name}'}})"
                    query += "\nCREATE (action)-[:USES]->(instrument)"
                if node_source_name:
                    query += f"\nWITH action MATCH (source {{name: '{node_source_name}'}})"
                    query += "\nCREATE (action)-[:FROM]->(source)"
                if node_target_name:
                    query += f"\nWITH action MATCH (target {{name: '{node_target_name}'}})"
                    query += "\nCREATE (action)-[:TO]->(target)"
                if prev_action_id:
                    query += f"\nWITH action MATCH (prev_action:Activity {{action_id: '{prev_action_id}'}})"
                    query += "\nCREATE (prev_action)-[:NEXT_ACTION]->(action)"

                query += "\nRETURN action.action_id AS action_id"

                result = session.run(query)
                record = result.single()
                return record["action_id"] if record else False
            except Exception as e:
                print(f"Error adding action: {e}")
                return False


    def compose_action_id(self, action_name, time_range, node_agent_name, node_patient_name=None):
        action_id = f"{node_agent_name}_{action_name}_{node_patient_name}_{time_range}"
        return action_id

    def get_actions_related_to_entity(self, entity_name):
        """
        Get all actions related to an entity and sort them by action-chain order.

        :param entity_name: Entity name
        :return: List of actions sorted by the linked-list order
        """
        with self.driver.session(database=self.database) as session:
            try:
                # Query all action nodes directly related to the entity.
                query = """
                MATCH (entity {name: $entity_name})

                // Entity as the performer of an action
                OPTIONAL MATCH (entity)-[:PERFORMS]->(action1:Activity)

                // Entity as the patient/receiver of an action
                OPTIONAL MATCH (action2:Activity)-[:AFFECTS]->(entity)

                // Entity as the instrument/tool of an action
                OPTIONAL MATCH (action3:Activity)-[:USES]->(entity)

                // Entity as the source of an action
                OPTIONAL MATCH (action4:Activity)-[:FROM]->(entity)

                // Entity as the target of an action
                OPTIONAL MATCH (action5:Activity)-[:TO]->(entity)

                // Merge all related actions
                WITH collect(distinct action1) + collect(distinct action2) +
                     collect(distinct action3) + collect(distinct action4) +
                     collect(distinct action5) as actions

                UNWIND actions as action
                WITH action
                WHERE action IS NOT NULL
                RETURN DISTINCT action
                ORDER BY action.time_range
                """

                result = session.run(query, entity_name=entity_name)
                actions = [dict(record["action"]) for record in result]

                # If there are no related actions, return an empty list.
                if not actions:
                    return []

                # Organize actions by linked-list order.
                # 1. Find actions with no predecessor as chain heads.
                head_actions = [action for action in actions if
                                "prev_action_id" not in action or action["prev_action_id"] is None]

                # 2. Build full chains starting from each head.
                ordered_chains = []
                for head in head_actions:
                    chain = [head]
                    visited_ids = {head.get("action_id")}
                    current_id = head.get("next_action_id")

                    # Traverse the chain.
                    while current_id:
                        next_action = next((action for action in actions if action.get("action_id") == current_id),
                                           None)

                        # Stop if the next action is missing or a cycle is detected.
                        if not next_action or next_action.get("action_id") in visited_ids:
                            break

                        chain.append(next_action)
                        visited_ids.add(next_action.get("action_id"))
                        current_id = next_action.get("next_action_id")

                    ordered_chains.append(chain)

                # 3. Handle isolated actions (neither heads nor included in any chain).
                included_ids = {node.get("action_id") for chain in ordered_chains for node in chain}
                isolated_actions = [action for action in actions if action.get("action_id") not in included_ids]

                # Treat each isolated action as its own chain.
                for action in isolated_actions:
                    ordered_chains.append([action])

                # 4. Sort chains by time.
                ordered_chains.sort(key=lambda chain: chain[0].get("time_range", ""))

                # 5. Flatten into a single list.
                flat_actions = [action for chain in ordered_chains for action in chain]

                # 6. Ensure each action has an action_name field.
                for action in flat_actions:
                    if "name" in action and "action_name" not in action:
                        action["action_name"] = action["name"]

                return flat_actions

            except Exception as e:
                print(f"Error getting actions related to entity: {e}")
                import traceback
                traceback.print_exc()
                return []

    def get_action_with_relations(self, action_id):
        """
        Get an action node by action_id and all associated nodes/relationships.

        :param action_id: Unique action identifier
        :return: Dict containing the action node and its relations in the form:
                 {
                    'action': dict,
                    'relations': [
                        {
                            'type': str,
                            'direction': 'outgoing'/'incoming',
                            'node': dict,
                            'props': dict
                        },
                        ...
                    ],
                    'prev_action': Previous action node properties (if any),
                    'next_action': Next action node properties (if any)
                 }
        """
        with self.driver.session(database=self.database) as session:
            try:
                # Query the action node.
                action_query = """
                MATCH (action:Activity {action_id: $action_id})
                RETURN action
                """
                result = session.run(action_query, action_id=action_id)
                record = result.single()
                if not record:
                    print(f"No node found for action_id={action_id}")
                    return None

                action_node = dict(record["action"])

                # Add action_name field for compatibility.
                if "name" in action_node and "action_name" not in action_node:
                    action_node["action_name"] = action_node["name"]

                # Query all relationships and related nodes for this action.
                relations_query = """
                MATCH (action:Activity {action_id: $action_id})
                OPTIONAL MATCH (action)-[rel1]->(related)
                OPTIONAL MATCH (source)-[rel2]->(action)
                RETURN 
                    collect(DISTINCT {
                        type: type(rel1), 
                        direction: 'outgoing', 
                        node: related, 
                        props: properties(rel1)
                    }) AS outgoing,
                    collect(DISTINCT {
                        type: type(rel2), 
                        direction: 'incoming', 
                        node: source, 
                        props: properties(rel2)
                    }) AS incoming
                """

                rel_result = session.run(relations_query, action_id=action_id)
                rel_record = rel_result.single().data()

                # Merge outgoing and incoming relations.
                relations = []

                # Process outgoing relations.
                if rel_record and "outgoing" in rel_record:
                    for rel in rel_record["outgoing"]:
                        if rel["node"] is not None:  # Filter out null nodes.
                            relations.append({
                                "type": rel["type"],
                                "direction": rel["direction"],
                                "node": dict(rel["node"]),
                                "props": rel["props"]
                            })

                # Process incoming relations.
                if rel_record and "incoming" in rel_record:
                    for rel in rel_record["incoming"]:
                        if rel["node"] is not None:  # Filter out null nodes.
                            relations.append({
                                "type": rel["type"],
                                "direction": rel["direction"],
                                "node": dict(rel["node"]),
                                "props": rel["props"]
                            })

                # Get previous and next actions.
                chain_query = """
                MATCH (action:Activity {action_id: $action_id})
                OPTIONAL MATCH (prev:Activity {action_id: action.prev_action_id})
                OPTIONAL MATCH (next:Activity {next_action_id: $action_id})
                RETURN prev, next
                """

                chain_result = session.run(chain_query, action_id=action_id)
                chain_record = chain_result.single()

                result_dict = {
                    "action": action_node,
                    "relations": relations,
                    "prev_action": dict(chain_record["prev"]) if chain_record and chain_record["prev"] else None,
                    "next_action": dict(chain_record["next"]) if chain_record and chain_record["next"] else None
                }

                # Fix action_name field for prev/next nodes.
                if result_dict["prev_action"] and "name" in result_dict["prev_action"] and "action_name" not in \
                        result_dict["prev_action"]:
                    result_dict["prev_action"]["action_name"] = result_dict["prev_action"]["name"]

                if result_dict["next_action"] and "name" in result_dict["next_action"] and "action_name" not in \
                        result_dict["next_action"]:
                    result_dict["next_action"]["action_name"] = result_dict["next_action"]["name"]

                return result_dict

            except Exception as e:
                print(f"Error getting action and relations: {e}")
                import traceback
                traceback.print_exc()
                return None

    def get_action_chain(self, start_action_id=None):
        """
        Get an action chain. If start_action_id is provided, return the chain starting there;
        otherwise return all chains that start from actions with no predecessor.

        :param start_action_id: Optional starting action id
        :return: Actions ordered by linked-list relations
        """
        with self.driver.session(database=self.database) as session:
            try:
                # If a starting action id is provided...
                if start_action_id:
                    # First fetch the starting action.
                    query = """
                    MATCH (start:Activity {action_id: $start_id})
                    RETURN start
                    """
                    result = session.run(query, start_id=start_action_id)
                    start_node = result.single()

                    if not start_node:
                        return []

                    # Then build the action chain.
                    action_chain = [dict(start_node["start"])]
                    current_id = start_node["start"].get("next_action_id")

                    # Traverse until the linked list ends.
                    while current_id:
                        query = """
                        MATCH (action:Activity {action_id: $action_id})
                        OPTIONAL MATCH (agent)-[:PERFORMS]->(action)
                        RETURN action, agent.name as agent_name
                        """
                        result = session.run(query, action_id=current_id)
                        record = result.single()

                        if not record:
                            break

                        action_data = dict(record["action"])
                        action_data["agent_name"] = record["agent_name"]
                        action_chain.append(action_data)

                        current_id = action_data.get("next_action_id")

                    return action_chain

                else:
                    # If no starting id is provided, find all actions with no predecessor.
                    query = """
                    MATCH (start:Activity)
                    WHERE start.prev_action_id IS NULL
                    RETURN start
                    ORDER BY start.time_range
                    """
                    result = session.run(query)

                    all_chains = []

                    for record in result:
                        start_node = dict(record["start"])
                        # Get full chain for each head node.
                        chain = [start_node]
                        current_id = start_node.get("next_action_id")

                        while current_id:
                            sub_query = """
                            MATCH (action:Activity {action_id: $action_id})
                            OPTIONAL MATCH (agent)-[:PERFORMS]->(action)
                            RETURN action, agent.name as agent_name
                            """
                            sub_result = session.run(sub_query, action_id=current_id)
                            sub_record = sub_result.single()

                            if not sub_record:
                                break

                            action_data = dict(sub_record["action"])
                            action_data["agent_name"] = sub_record["agent_name"]
                            chain.append(action_data)

                            current_id = action_data.get("next_action_id")

                        all_chains.append(chain)

                    return all_chains

            except Exception as e:
                print(f"Error getting action chain: {e}")
                return []

    def get_actions_in_period(self, time_range, agent_name=None):
        """
        Get all actions within a time range (optionally filtered by agent), ordered by the action chain.

        :param time_range: Time range in hh:mm:ss-hh:mm:ss
        :param agent_name: Optional agent name
        :return: List of actions ordered by linked-list relations
        """
        with self.driver.session(database=self.database) as session:
            try:
                # Build base query.
                # Find all action nodes within the specified time range.
                query_parts = [
                    "MATCH (action:Activity)",
                    "WHERE",
                    # Use simple string comparisons to approximate overlap.
                    # The action's time range must intersect with the given time range.
                    "(action.time_range STARTS WITH $time_range OR action.time_range CONTAINS $time_range",
                    "OR $time_range STARTS WITH SUBSTRING(action.time_range, 0, 8)",
                    "OR $time_range CONTAINS SUBSTRING(action.time_range, 0, 8))"
                ]

                # If an agent is specified, add the constraint.
                if agent_name:
                    query_parts.append("WITH action")
                    query_parts.append("MATCH (agent:Person {name: $agent_name})-[:PERFORMS]->(action)")

                # Return all matching actions.
                query_parts.append("RETURN action ORDER BY action.time_range")

                query = "\n".join(query_parts)

                params = {"time_range": time_range}
                if agent_name:
                    params["agent_name"] = agent_name

                result = session.run(query, params)

                # Collect action nodes.
                action_nodes = [dict(record["action"]) for record in result]

                # If there are no actions, return an empty list.
                if not action_nodes:
                    return []

                # Find head nodes (nodes with no predecessor).
                head_nodes = [node for node in action_nodes if
                              "prev_action_id" not in node or node["prev_action_id"] is None]

                # Build all chains.
                chains = []
                for head in head_nodes:
                    chain = [head]
                    current_id = head.get("next_action_id")

                    # Track visited node ids to avoid cycles.
                    visited_ids = {head.get("action_id")}

                    # Build the chain by linked-list pointers.
                    while current_id:
                        # Find the next node within the current set.
                        next_node = next((node for node in action_nodes
                                          if node.get("action_id") == current_id), None)

                        # Stop if missing or already visited.
                        if not next_node or next_node.get("action_id") in visited_ids:
                            break

                        chain.append(next_node)
                        visited_ids.add(next_node.get("action_id"))
                        current_id = next_node.get("next_action_id")

                    chains.append(chain)

                # Handle isolated actions (neither heads nor included in any chain).
                included_ids = {node.get("action_id") for chain in chains for node in chain}
                isolated_nodes = [node for node in action_nodes
                                  if node.get("action_id") not in included_ids]

                # Treat each isolated node as a separate chain.
                for node in isolated_nodes:
                    chains.append([node])

                # Sort all chains by time.
                chains.sort(key=lambda chain: chain[0].get("time_range", ""))

                # Flatten nested chains into a single list.
                flat_actions = [node for chain in chains for node in chain]

                # Add action_name property if missing.
                for action in flat_actions:
                    if "name" in action and "action_name" not in action:
                        action["action_name"] = action["name"]

                return flat_actions

            except Exception as e:
                print(f"Error getting actions in period: {e}")
                return []

    def extract_subgraph_by_nodes(self, node_names, max_path_length=10):
        """
        Extract a subgraph around a set of key nodes.

        Includes:
        - All key nodes
        - Nodes/relationships on paths connecting key nodes
        - Nodes/relationships directly connected to key nodes
        - For Activity nodes, all nodes/relationships connected to the activity

        The returned dict includes key nodes, other nodes, activities, paths, and relationships.

        :param node_names: List of key node names
        :return: Subgraph data dict
        """
        if not node_names:
            return {
                "key_nodes": [],
                "other_nodes": [],
                "activities": [],
                "paths": [],
                "relationships": []
            }

        # Collect all data.
        key_nodes = self._get_nodes_info(node_names)
        paths_data = self._get_paths_between_key_nodes(node_names, max_path_length=max_path_length)
        direct_connections = self._get_direct_connections(node_names)

        # Initialize result collections.
        activity_nodes = []
        activity_names = set()  # Track activity node names already added.
        other_nodes = []
        activity_relationships = []
        other_relationships = []

        # Process path data.
        for path in paths_data:
            for item in path:
                if "node" in item:
                    node_data = item["node"]
                    node_label = item.get("label")

                    # Whether this is an activity node.
                    if node_label == NodeLabels.ACTIVITY.value:
                        node_name = node_data.get("name")
                        if node_name not in activity_names:
                            activity_names.add(node_name)
                            activity_nodes.append(node_data)
                            # Get relationships connected to the activity.
                            action_rels = self._get_activity_relationships(node_name)
                            activity_relationships.extend(action_rels)
                    else:
                        # Avoid adding key nodes into other_nodes.
                        if not any(n.get("name") == node_data.get("name") for n in key_nodes) and \
                                not any(n.get("name") == node_data.get("name") for n in other_nodes):
                            other_nodes.append(node_data)

                # Process relationships.
                if "relationship" in item:
                    rel_data = item["relationship"]
                    rel_type = item.get("type")
                    start_node = item.get("start_node", {}).get("name", "")
                    end_node = item.get("end_node", {}).get("name", "")

                    # Create an enriched relationship dict.
                    enriched_rel = dict(rel_data)
                    enriched_rel.update({
                        "type": rel_type,
                        "start_node": start_node,
                        "end_node": end_node
                    })

                    # Check whether this is an action relationship.
                    if self.is_action_relation(rel_type):
                        if not any(r.get("id") == enriched_rel.get("id") for r in activity_relationships):
                            activity_relationships.append(enriched_rel)
                    else:
                        if not any(r.get("id") == enriched_rel.get("id") for r in other_relationships):
                            other_relationships.append(enriched_rel)

        # Process direct connections.
        for conn in direct_connections:
            if conn["node_type"] == NodeLabels.ACTIVITY.value:
                node_name = conn["node"].get("name")
                if node_name not in activity_names:
                    activity_names.add(node_name)
                    activity_nodes.append(conn["node"])
                    # Get relationships connected to the activity.
                    action_rels = self._get_activity_relationships(node_name)
                    activity_relationships.extend(action_rels)
            else:
                if not any(n.get("name") == conn["node"].get("name") for n in key_nodes) and \
                        not any(n.get("name") == conn["node"].get("name") for n in other_nodes):
                    other_nodes.append(conn["node"])

            # Process relationship.
            if "relationship" in conn:
                rel_data = conn["relationship"]
                rel_type = conn["relation_type"]
                start_node = conn.get("source_node", "")
                end_node = conn.get("target_node", "")

                # Create an enriched relationship dict.
                enriched_rel = dict(rel_data)
                enriched_rel.update({
                    "type": rel_type,
                    "start_node": start_node,
                    "end_node": end_node
                })

                # Check whether this is an action relationship.
                if self.is_action_relation(rel_type):
                    if not any(r.get("id") == enriched_rel.get("id") for r in activity_relationships):
                        activity_relationships.append(enriched_rel)
                else:
                    if not any(r.get("id") == enriched_rel.get("id") for r in other_relationships):
                        other_relationships.append(enriched_rel)

        # Sort activity nodes by the linked-list order.
        ordered_activities = self._order_activities_by_chain(activity_nodes)

        # Build return result.
        return {
            "key_nodes": key_nodes,
            "other_nodes": other_nodes,
            "activities": ordered_activities,
            "paths": paths_data,
            "relationships": {
                "activity_relationships": activity_relationships,
                "other_relationships": other_relationships
            }
        }

    def format_subgraph_json(self, subgraph):
        """
        Format a scene-graph subgraph to a JSON string.

        :param subgraph: Subgraph returned by extract_subgraph_by_nodes
        :return: JSON string
        """
        result = {
            "key_nodes": [],
            "other_nodes": [],
            "activities": [],
            "paths": [],
            "relationships": {}
        }

        # Format key nodes.
        if subgraph["key_nodes"]:
            for node in subgraph["key_nodes"]:
                result["key_nodes"].append({
                    "label": node.get("labels", "Unknown Type"),
                    "name": node.get("name", "Unnamed"),
                    "info": node.get("info", ""),
                    "time_range": node.get("time_range", "")
                })

        # Format other nodes.
        if subgraph["other_nodes"]:
            for node in subgraph["other_nodes"]:
                result["other_nodes"].append({
                    "label": node.get("labels", "Unknown Type"),
                    "name": node.get("name", "Unnamed"),
                    "info": node.get("info", "No description"),
                    "time_range": node.get("time_range", "")
                })

        # Format activities.
        if subgraph["activities"]:
            for i, action in enumerate(subgraph["activities"]):
                result["activities"].append({
                    "action_id": action.get("action_id", "No id"),
                    "name": action.get("name", "Unnamed Action"),
                    "info": action.get("info", "No description"),
                    "time_range": action.get("time_range", "No time information"),
                    "agent": action.get("agent_name", ""),
                    "patient": action.get("patient_name", ""),
                    "instrument": action.get("instrument_name", ""),
                    "source": action.get("source_name", ""),
                    "target": action.get("target_name", "")
                })

        # Format paths.
        if subgraph["paths"]:
            for i, path in enumerate(subgraph["paths"]):
                path_data = {
                    "id": i + 1,
                    "items": []
                }

                for item in path:
                    if item["type"] == 'node':
                        path_data["items"].append({
                            "type": "node",
                            "name": item["name"],
                            "label": item["label"]
                        })
                    else:
                        path_data["items"].append({
                            "type": "relationship",
                            "relation_type": item["relation_type"],
                            "start_time": item['properties'].get('start_time', ''),
                            "end_time": item['properties'].get('end_time', '')
                        })

                result["paths"].append(path_data)

        # Format relationships.
        if "relationships" in subgraph:
            result["relationships"] = {
                "activity_relationships": [],
                "other_relationships": []
            }

            # Add other relationships.
            if subgraph["relationships"].get("other_relationships"):
                for rel in subgraph["relationships"]["other_relationships"]:
                    result["relationships"]["other_relationships"].append({
                        "type": rel.get("type", "Unknown Relationship"),
                        "start_node": {
                            "name": rel.get("start_node", {}).get("name", "")
                        },
                        "end_node": {
                            "name": rel.get("end_node", {}).get("name", "")
                        },
                        "info": rel.get("info", ""),
                        "start_time": rel.get("start_time", ""),
                        "end_time": rel.get("end_time", "")
                    })

        # Return formatted JSON string.
        return json.dumps(result, ensure_ascii=False, indent=2)

    def format_subgraph(self, subgraph):
        """
        Format the scene graph subgraph and return it as a text string
        :param subgraph: Subgraph data returned by the extract_subgraph_by_nodes function
        :return: Formatted subgraph text string
        """
        output = []
        output.append("\n======= Scene Graph Subgraph =======\n")
        # If there are no key nodes, don't output
        if not subgraph["key_nodes"]:
            output.append("   No key nodes")
            return "\n".join(output)

        # Add key node information
        output.append("[Key Nodes]")
        if subgraph["key_nodes"]:
            for i, node in enumerate(subgraph["key_nodes"]):
                node_label = node.get("labels", "Unknown Type")
                node_name = node.get("name", "Unnamed")
                node_info = node.get("info", None)
                time_range = node.get("time_range", "")

                time_info = f" ({time_range})" if time_range else ""
                output.append(f"{i + 1}. {node_name}{time_info}")
                if node_info:
                    output.append(f"   Description: {node_info}")
        else:
            output.append("   No key nodes")

        output.append("\n[Other Nodes]")
        if subgraph["other_nodes"]:
            for i, node in enumerate(subgraph["other_nodes"]):
                node_label = node.get("labels", "Unknown Type")
                node_name = node.get("name", "Unnamed")
                node_info = node.get("info", "No description")
                time_range = node.get("time_range", "")

                time_info = f" ({time_range})" if time_range else ""
                output.append(f"{i + 1}. {node_name}{time_info}")
                output.append(f"   Description: {node_info}")
        else:
            output.append("   No other nodes")
        """
        output.append("\n【Action Sequence】")
        if subgraph["activities"]:
            for i, action in enumerate(subgraph["activities"]):
                action_name = action.get("name", "Unnamed Action")
                action_info = action.get("info", "No description")
                time_range = action.get("time_range", "No time information")
                agent_name = action.get("agent_name", "")


                # Get other related entities
                patient_name = action.get("patient_name", "")
                instrument_name = action.get("instrument_name", "")
                source_name = action.get("source_name", "")
                target_name = action.get("target_name", "")

                output.append(f"{i + 1}. {action_name} ({time_range})")
                output.append(f"   Description: {action_info}")

                # Add all related entities
                # if agent_name:
                #     output.append(f"   Agent: {agent_name}")
                # if patient_name:
                #     output.append(f"   Patient: {patient_name}")
                # if instrument_name:
                #     output.append(f"   Instrument: {instrument_name}")
                # if source_name:
                #     output.append(f"   Source: {source_name}")
                # if target_name:
                #     output.append(f"   Target: {target_name}")
        else:
            output.append("   No actions")
        """

        output.append("\n【Paths Between Key Nodes】")
        if subgraph["paths"]:
            # Use a set to deduplicate identical path text.
            unique_path_texts = set()
            path_count = 0

            for i, path in enumerate(subgraph["paths"]):
                # Build a text representation of the path.
                path_text = ""
                for j, item in enumerate(path):
                    if item["type"] == 'node':
                        node_name = item["name"]
                        path_text += f"{node_name}"
                    else:
                        rel_type = item["relation_type"]
                        # Only show time range when start_time exists.
                        if item['properties'].get('start_time'):
                            path_text += f"--{rel_type}({item['properties'].get('start_time', 'xx')}-{item['properties'].get('end_time', 'xx')})->"
                        else:
                            # Otherwise, only show the relation type.
                            path_text += f"--{rel_type}-->"

                # Skip if already output.
                if path_text not in unique_path_texts:
                    path_count += 1
                    unique_path_texts.add(path_text)
                    output.append(f"Path {path_count}:")
                    output.append(f"   {path_text}")

            # If no unique paths after deduplication.
            if path_count == 0:
                output.append("   No unique paths")
        else:
            output.append("   No paths")

        output.append("\n[Relationships]")
        if "relationships" in subgraph:
            # Add action relationships
            # output.append("Action Relationships:")
            # if subgraph["relationships"].get("activity_relationships"):
            #     for i, rel in enumerate(subgraph["relationships"]["activity_relationships"]):
            #         rel_type = rel.get("type", "Unknown Relationship")
            #         start_node = rel.get("start_node", "")
            #         end_node = rel.get("end_node", "")
            #         info = rel.get("info", None)
            #
            #         output.append(f"{i + 1}. {start_node} --{rel_type}-> {end_node}")
            #         if info:
            #             output.append(f"   Description: {info}")
            # else:
            #     output.append("   No action relationships")

            # Add other relationships
            # output.append("\nOther Relationships:")
            if subgraph["relationships"].get("other_relationships"):
                for i, rel in enumerate(subgraph["relationships"]["other_relationships"]):
                    rel_type = rel.get("type", "Unknown Relationship")
                    start_node = rel.get("start_node", "")
                    end_node = rel.get("end_node", "")
                    info = rel.get("info", None)
                    start_time = rel.get("start_time", "")
                    end_time = rel.get("end_time", "")

                    output.append(
                        f"{i + 1}. {start_node['name']} --{rel_type}({start_time}-{end_time})-> {end_node['name']}")
                    if info:
                        output.append(f"   Description: {info}")
            else:
                output.append("   No other relationships")
        else:
            output.append("   No relationship data")

        output.append("\n======= End of Subgraph =======")

        # Combine all lines into a single string and return
        return "\n".join(output)

    def _order_activities_by_chain(self, activity_nodes):
        """
        Order activity nodes by their linked-list relations.

        :param activity_nodes: List of activity node dicts
        :return: Ordered list of activity nodes
        """
        if not activity_nodes:
            return []

        # Build action_id -> node mapping.
        activity_map = {}
        for node in activity_nodes:
            action_id = node.get("action_id")
            if action_id:
                activity_map[action_id] = node

        # Head nodes are those without a predecessor.
        head_nodes = [node for node in activity_nodes if
                      "prev_action_id" not in node or not node.get("prev_action_id")]

        # Build all chains.
        chains = []
        for head in head_nodes:
            chain = [head]
            current_id = head.get("next_action_id")

            # Track visited ids to avoid cycles.
            visited_ids = {head.get("action_id")}

            # Follow next pointers.
            while current_id and current_id in activity_map:
                next_node = activity_map[current_id]

                # Avoid cycles.
                if next_node.get("action_id") in visited_ids:
                    break

                chain.append(next_node)
                visited_ids.add(next_node.get("action_id"))
                current_id = next_node.get("next_action_id")

            chains.append(chain)

        # Handle isolated nodes (not in any chain).
        included_ids = {node.get("action_id") for chain in chains for node in chain}
        isolated_nodes = [node for node in activity_nodes
                          if node.get("action_id") not in included_ids]

        # Put each isolated node into its own chain.
        for node in isolated_nodes:
            chains.append([node])

        # Sort chains by time.
        chains.sort(key=lambda chain: chain[0].get("time_range", ""))

        # Flatten.
        ordered_activities = [node for chain in chains for node in chain]

        return ordered_activities

    def _get_nodes_info(self, node_names):
        """
        Get node properties for a list of names.

        :param node_names: List of node names
        :return: List of node dicts
        """
        nodes = []
        with self.driver.session(database=self.database) as session:
            for name in node_names:
                query = """
                MATCH (n {name: $name})
                RETURN n, labels(n) as labels
                """
                result = session.run(query, name=name)
                for record in result:
                    node_data = dict(record["n"])
                    node_data["labels"] = record["labels"]
                    nodes.append(node_data)
        return nodes

    def _get_paths_between_key_nodes(self, node_names, max_path_length=5):
        """
        Get shortest paths between key nodes while respecting direction.

        :param node_names: List of key node names
        :param max_path_length: Maximum path length
        :return: List of path data; each path follows relationship direction
        """
        paths = []
        processed_pairs = set()
        node_names = list(node_names)

        with self.driver.session(database=self.database) as session:
            # Get paths between all node pairs.
            for i, source in enumerate(node_names):
                for target in node_names[i + 1:]:
                    # Ensure the two nodes are different.
                    if source == target:
                        continue

                    if (source, target) in processed_pairs:
                        continue

                    processed_pairs.add((source, target))

                    # Query directed path (source->target).
                    directed_query = f"""
                    MATCH path = shortestPath((source {{name: $source}})-[*1..{max_path_length}]->(target {{name: $target}}))
                    RETURN path
                    """
                    result = session.run(directed_query, source=source, target=target)

                    for record in result:
                        path = record["path"]
                        path_data = []

                        # Process nodes and relationships in the path.
                        nodes = path.nodes
                        relationships = path.relationships

                        # Add the first node.
                        start_node = dict(nodes[0])
                        start_labels = list(nodes[0].labels)

                        path_data.append({
                            "type": "node",
                            "name": start_node.get("name", ""),
                            "label": start_labels[0] if start_labels else "Unknown",
                            "properties": start_node
                        })

                        # Add relationships and subsequent nodes.
                        for i in range(len(relationships)):
                            rel = relationships[i]
                            end_node = nodes[i + 1]

                            # Relationship info.
                            rel_dict = dict(rel)
                            rel_type = rel.type

                            # Add relationship.
                            path_data.append({
                                "type": "relationship",
                                "relation_type": rel_type,
                                "direction": "outgoing",  # always outgoing for this constructed path
                                "properties": rel_dict,
                                "source": dict(rel.nodes[0]).get("name", ""),
                                "target": dict(rel.nodes[1]).get("name", "")
                            })

                            # Add target node.
                            end_node_dict = dict(end_node)
                            end_labels = list(end_node.labels)

                            path_data.append({
                                "type": "node",
                                "name": end_node_dict.get("name", ""),
                                "label": end_labels[0] if end_labels else "Unknown",
                                "properties": end_node_dict
                            })

                        if path_data:
                            paths.append(path_data)

                    # Query reverse path (target->source).
                    reverse_query = f"""
                    MATCH path = shortestPath((target {{name: $target}})-[*1..{max_path_length}]->(source {{name: $source}}))
                    RETURN path
                    """
                    result = session.run(reverse_query, source=source, target=target)

                    for record in result:
                        path = record["path"]
                        path_data = []

                        # Process nodes and relationships in the path.
                        nodes = path.nodes
                        relationships = path.relationships

                        # Add the first node.
                        start_node = dict(nodes[0])
                        start_labels = list(nodes[0].labels)

                        path_data.append({
                            "type": "node",
                            "name": start_node.get("name", ""),
                            "label": start_labels[0] if start_labels else "Unknown",
                            "properties": start_node
                        })

                        # Add relationships and subsequent nodes.
                        for i in range(len(relationships)):
                            rel = relationships[i]
                            end_node = nodes[i + 1]

                            # Relationship info.
                            rel_dict = dict(rel)
                            rel_type = rel.type

                            # Add relationship.
                            path_data.append({
                                "type": "relationship",
                                "relation_type": rel_type,
                                "direction": "outgoing",  # always outgoing for this constructed path
                                "properties": rel_dict,
                                "source": dict(rel.nodes[0]).get("name", ""),
                                "target": dict(rel.nodes[1]).get("name", "")
                            })

                            # Add target node.
                            end_node_dict = dict(end_node)
                            end_labels = list(end_node.labels)

                            path_data.append({
                                "type": "node",
                                "name": end_node_dict.get("name", ""),
                                "label": end_labels[0] if end_labels else "Unknown",
                                "properties": end_node_dict
                            })

                        if path_data:
                            paths.append(path_data)

        return paths

    def _get_direct_connections(self, node_names):
        """
        Get nodes/relationships directly connected to key nodes.

        :param node_names: List of key node names
        :return: List of direct connection dicts
        """
        connections = []

        with self.driver.session(database=self.database) as session:
            for name in node_names:
                # Get outgoing relationships.
                query = """
                MATCH (n {name: $name})-[r]->(m)
                RETURN n, r, m, labels(m) as m_labels
                """
                result = session.run(query, name=name)

                for record in result:
                    connections.append({
                        "source_node": dict(record["n"]),
                        "relationship": dict(record["r"]),
                        "relation_type": record["r"].type,
                        "node": dict(record["m"]),
                        "target_node": dict(record["m"]),
                        "node_type": record["m_labels"][0] if record["m_labels"] else None,
                        "direction": "outgoing"
                    })

                # Get incoming relationships.
                query = """
                MATCH (m)-[r]->(n {name: $name})
                RETURN n, r, m, labels(m) as m_labels
                """
                result = session.run(query, name=name)

                for record in result:
                    connections.append({
                        "target_node": dict(record["n"]),
                        "relationship": dict(record["r"]),
                        "relation_type": record["r"].type,
                        "node": dict(record["m"]),
                        "source_node": dict(record["m"]),
                        "node_type": record["m_labels"][0] if record["m_labels"] else None,
                        "direction": "incoming"
                    })

        return connections

    def _get_activity_relationships(self, activity_name):
        """
        Get all relationships connected to an activity node.

        :param activity_name: Activity node name
        :return: List of relationship dicts
        """
        with self.driver.session(database=self.database) as session:
            try:
                # Query all relationships connected to the activity (both directions).
                query = """
                MATCH (act:Activity {name: $activity_name})

                // Outgoing relations
                OPTIONAL MATCH (act)-[out_rel]->(out_node)

                // Incoming relations
                OPTIONAL MATCH (in_node)-[in_rel]->(act)

                // Merge results with relationship type and node info
                RETURN 
                    collect({
                        relationship: properties(out_rel),
                        type: type(out_rel),
                        start_node: act.name,
                        end_node: out_node.name,
                        id: id(out_rel)
                    }) AS outgoing,
                    collect({
                        relationship: properties(in_rel),
                        type: type(in_rel),
                        start_node: in_node.name,
                        end_node: act.name,
                        id: id(in_rel)
                    }) AS incoming
                """

                result = session.run(query, activity_name=activity_name)
                record = result.single()

                # Merge outgoing and incoming.
                relationships = []

                # Process outgoing relations.
                for rel in record["outgoing"]:
                    if rel["relationship"] is not None:  # Filter out null relationships.
                        # Create an enriched relationship dict.
                        enriched_rel = dict(rel["relationship"])  # Copy original properties.
                        enriched_rel.update({
                            "type": rel["type"],
                            "start_node": rel["start_node"],
                            "end_node": rel["end_node"],
                            "id": rel["id"]
                        })
                        relationships.append(enriched_rel)

                # Process incoming relations.
                for rel in record["incoming"]:
                    if rel["relationship"] is not None:  # Filter out null relationships.
                        # Create an enriched relationship dict.
                        enriched_rel = dict(rel["relationship"])  # Copy original properties.
                        enriched_rel.update({
                            "type": rel["type"],
                            "start_node": rel["start_node"],
                            "end_node": rel["end_node"],
                            "id": rel["id"]
                        })
                        relationships.append(enriched_rel)

                return relationships

            except Exception as e:
                print(f"Error getting activity relationships: {e}")
                import traceback
                traceback.print_exc()
                return []

    def is_action_relation(self, relation_type):
        """
        Check whether a relationship type is an action relation.

        :param relation_type: Relationship type
        :return: True if it is an action relation
        """
        action_relations = ["PERFORMS", "AFFECTS", "USES", "FROM", "TO"]
        return relation_type in action_relations

    def clear_database(self):
        """
        Clear the database: nodes, relationships, constraints, and indexes.
        """
        with self.driver.session(database=self.database) as session:
            try:
                # Drop all constraints and indexes.
                constraints_query = "SHOW CONSTRAINTS"
                constraints_result = session.run(constraints_query)

                # Drop each constraint.
                for record in constraints_result:
                    constraint_name = record.get("name")
                    if constraint_name:
                        session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")

                # Drop all indexes.
                indexes_query = "SHOW INDEXES"
                indexes_result = session.run(indexes_query)

                # Drop each index.
                for record in indexes_result:
                    index_name = record.get("name")
                    if index_name:
                        session.run(f"DROP INDEX {index_name} IF EXISTS")

                # Delete all nodes and relationships.
                session.run("MATCH (n) DETACH DELETE n")

                print("Database cleared")
                return True
            except Exception as e:
                print(f"Error clearing database: {e}")
                return False

    def get_all_actions_in_time_range(self, time_range: str):
        """
        Get all action names within a time range and order them by the action chain.

        :param time_range: Time range in "hh:mm:ss-hh:mm:ss"
        :return: List of action names ordered by linked-list relations
        """
        with self.driver.session(database=self.database) as session:
            try:
                # Query all action nodes in the specified time range.
                query = """
                MATCH (action:Activity)
                WHERE action.time_range = $time_range OR action.time_range CONTAINS $time_range 
                OR $time_range CONTAINS action.time_range
                RETURN action ORDER BY action.time_range
                """

                result = session.run(query, time_range=time_range)

                # Collect all action nodes.
                action_nodes = [dict(record["action"]) for record in result]

                # If there are no actions, return an empty list.
                if not action_nodes:
                    return []

                # Find head nodes (nodes with no predecessor).
                head_nodes = [node for node in action_nodes if
                              "prev_action_id" not in node or not node.get("prev_action_id")]

                # Build all chains.
                chains = []
                for head in head_nodes:
                    chain = [head]
                    current_id = head.get("next_action_id")
                    # Track visited node ids to avoid cycles.
                    visited_ids = {head.get("action_id")}

                    # Build the chain by linked-list pointers.
                    while current_id and any(node.get("action_id") == current_id for node in action_nodes):
                        next_node = next((node for node in action_nodes if node.get("action_id") == current_id), None)

                        # Avoid cycles.
                        if next_node.get("action_id") in visited_ids:
                            break

                        chain.append(next_node)
                        visited_ids.add(next_node.get("action_id"))
                        current_id = next_node.get("next_action_id")

                    chains.append(chain)

                # Handle isolated action nodes.
                included_ids = {node.get("action_id") for chain in chains for node in chain}
                isolated_nodes = [node for node in action_nodes
                                  if node.get("action_id") not in included_ids]

                # Treat each isolated node as a separate chain.
                for node in isolated_nodes:
                    chains.append([node])

                # Sort all chains by time.
                chains.sort(key=lambda chain: chain[0].get("time_range", ""))

                # Merge all chains into a single list.
                ordered_actions = [node.get("name", node.get("action_name", "Unnamed action"))
                                   for chain in chains for node in chain]

                return ordered_actions

            except Exception as e:
                print(f"Error getting actions in time range: {e}")
                import traceback
                traceback.print_exc()
                return []

    def count_triples(self):
        """
        Count and print the number of triples (relationships) in the current graph.

        :return: Triple count
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (a)-[r]->(b) RETURN count(r) AS triple_count")
            count = result.single()["triple_count"]
            print(f"Triple count in the current relation graph: {count}")
            return count



class NodeLabels(Enum):
    PERSON = "Person"
    OBJECT = "Object"
    ACTIVITY = "Activity"
    AREA = "Area"


