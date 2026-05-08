import json



def extract_json_from_markdown(text):
    """
    Extract JSON content from markdown-formatted text.
    Handles text like ```json {...} ``` by removing surrounding markdown markers and other explanatory text.
    If multiple JSON blocks exist, returns the last one.

    :param text: Text that may contain markdown-formatted JSON
    :return: Extracted JSON string, or the original text if nothing is found
    """
    import re

    # Find content between ```json and ```; collect all matches
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text)

    if matches:
        # Return the last extracted JSON block
        return matches[-1].strip()

    # If no match is found, return the original text
    return text

def parse_json_text(json_text):
    """
    Parse JSON-formatted text, first attempting to extract JSON from markdown.
    :param json_text: Text that may be markdown-formatted JSON
    :return: Parsed Python object
    """
    try:
        # First try extracting JSON from markdown
        clean_json = extract_json_from_markdown(json_text)
        # Ensure format specifiers in the JSON string do not cause issues
        parsed_data = json.loads(clean_json)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e} ")
        print("Error text: ", json_text)
        # Try escaping any format specifiers
        try:
            # Replace % with %% to avoid interpreting it as a format specifier
            escaped_json = clean_json.replace('%', '%%')
            parsed_data = json.loads(escaped_json)
            return parsed_data
        except:
            return None

def time_to_seconds(time_str):
    """
    Convert a time string in "mm:ss" format to seconds.
    :param time_str: A time string in "mm:ss" format
    :return: The corresponding number of seconds
    """
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def parse_label(label):
    """
    Parse labels in the form "id, type, description" into three separate values.
    """
    try:
        id_str, type_str, description_str = label.split(", ")
        id = int(id_str)
        type = type_str
        description = description_str
        return id, type, description
    except ValueError as e:
        print(f"Value error: {e}")
        return None, None, None

def check_nav_graph_format(json_obj):
    """
    Check whether the object matches the following JSON format:
    {
    "area_name": [
        {
            "start_end": [t1, t2],
            "description": "description of the period",
            "objects": [["id", "type", "description"], ...]
        }
    ],
    ...
}
    """
    if not isinstance(json_obj, dict):
        return False
    for area_name, area_info in json_obj.items():
        if not isinstance(area_info, list) or len(area_info) == 0:
            return False
        for info in area_info:
            if not isinstance(info, dict):
                return False
            if "start_end" not in info or "description" not in info or "objects" not in info:
                return False
            if not isinstance(info["start_end"], list) or len(info["start_end"]) != 2:
                return False
            if not isinstance(info["description"], str):
                return False
            if not isinstance(info["objects"], list):
                return False
    return True

def parse_time_range(text):
    """
    Parse a time range in the format "t1s-t2s" or "t1-t2s" into an integer list [t1, t2].
    """
    try:
        time_range = text.split('-')
        if len(time_range) != 2:
            raise ValueError("Invalid time range format")
        start_time = int(time_range[0].replace('s', ''))
        end_time = int(time_range[1].replace('s', ''))
        return [start_time, end_time]
    except ValueError as e:
        print(f"Time Value Error: {e}")
        return None
