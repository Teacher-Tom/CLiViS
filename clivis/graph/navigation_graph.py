"""Navigation graph over video periods, areas, objects, and activities."""

from clivis.graph.common import *

class NavigationGraph(object):
    def __init__(self, period_description_dict, video_path, seg_output_path, video_duration):
        """
        Initialize the navigation graph.

        :param period_description_dict: Video period description dict
        :param video_path: Path to the input video
        :param seg_output_path: Output directory/prefix for video segments
        :param video_duration: Video duration in seconds
        """
        # Reformat periods_info.
        self.periods_infos = {}
        periods_list = []
        self.periods_to_obj_names = {}  # period -> object names
        self.periods_to_activities = {}  # period -> activities
        self.periods_to_areas = {}
        self.person_names = set()
        self.area_names = set()
        self.obj_names = set()
        self.person_info = {}

        # Store descriptions that start beyond video duration (merged into the last valid period).
        out_of_range_descriptions = []
        last_valid_period = None

        for i, period in enumerate(period_description_dict["periods"]):
            # Convert start/end timestamps to seconds.
            start_time = utils.time_to_seconds(period['start_time'])
            end_time = utils.time_to_seconds(period['end_time'])

            # If the period starts after the video ends, collect its description for later merging.
            if start_time >= video_duration:
                out_of_range_descriptions.append(period['description'])
                continue

            # Clamp the time range to the video boundaries.
            if start_time < 0:
                start_time = 0
            if end_time > video_duration:
                end_time = video_duration
            if start_time >= end_time:
                continue

            # Convert back to hh:mm:ss.
            hours_start = int(start_time // 3600)
            minutes_start = int((start_time % 3600) // 60)
            seconds_start = int(start_time % 60)

            hours_end = int(end_time // 3600)
            minutes_end = int((end_time % 3600) // 60)
            seconds_end = int(end_time % 60)

            start_time = f"{hours_start:02}:{minutes_start:02}:{seconds_start:02}"
            end_time = f"{hours_end:02}:{minutes_end:02}:{seconds_end:02}"

            period_name = f"{start_time}-{end_time}"
            periods_list.append(period_name)
            self.periods_infos[period_name] = {
                "description": period["description"],
                "areas": [],
                "activities": [],
                "objects": []
            }
            # Initialize period -> objects mapping.
            self.periods_to_obj_names[period_name] = []

            # Track the last valid period.
            last_valid_period = period_name

        # Merge out-of-range descriptions into the last valid period.
        if out_of_range_descriptions and last_valid_period:
            combined_description = self.periods_infos[last_valid_period]["description"]
            for desc in out_of_range_descriptions:
                combined_description += " " + desc
            self.periods_infos[last_valid_period]["description"] = combined_description

        # Ensure the last valid period ends exactly at the video duration.
        if last_valid_period:
            # Parse start/end of the last valid period.
            last_start, last_end = last_valid_period.split("-")
            # Set the end time to the video duration.
            hours_end = int(video_duration // 3600)
            minutes_end = int((video_duration % 3600) // 60)
            seconds_end = int(video_duration % 60)
            new_end_time = f"{hours_end:02}:{minutes_end:02}:{seconds_end:02}"

            # If the end time differs, create a new period name and replace the old one.
            if last_end != new_end_time:
                new_period_name = f"{last_start}-{new_end_time}"
                # Copy original period info to the new period.
                self.periods_infos[new_period_name] = self.periods_infos[last_valid_period]
                # Update periods_list.
                periods_list[periods_list.index(last_valid_period)] = new_period_name
                # Update period -> objects mapping.
                if last_valid_period in self.periods_to_obj_names:
                    self.periods_to_obj_names[new_period_name] = self.periods_to_obj_names.pop(last_valid_period)
                # Delete the old period.
                del self.periods_infos[last_valid_period]
                last_valid_period = new_period_name

        # Split video by period list.
        self.video_segments_to_files = spilit_video.split_video(video_path, periods_list, output_prefix=seg_output_path)

        print("Video Divided: ", periods_list)

    def get_period_names(self):
        """
        Get all period names.
        :return: List of period names
        """
        return list(self.periods_infos.keys())

    def add_persons(self, persons):
        """
        Add person metadata.
        :param persons: List of person dicts
        """
        for person in persons:
            person_name = person["name"]
            person_info = person["info"]
            self.person_names.add(person_name)
            self.person_info[person_name] = person_info

    def get_person_info(self, person_name):
        """
        Get a person's info.
        :param person_name: Person name
        :return: Person info
        """
        return self.person_info.get(person_name, None)

    def add_areas(self, areas):
        """
        Add area metadata.
        :param areas: List of area dicts
        """
        for area in areas:
            area_name = area["name"]
            area_info = area["info"]
            time_range = area["time_range"]
            self.area_names.add(area_name)
            # If the area time range overlaps existing periods, assign it to those periods.
            time_range_list = spilit_video.find_time_range(time_range, self.periods_infos.keys())
            if time_range_list:
                for time_range in time_range_list:
                    if time_range in self.periods_infos:
                        self.periods_infos[time_range]["areas"].append(area_name)
                        if time_range not in self.periods_to_areas:
                            self.periods_to_areas[time_range] = []
                        self.periods_to_areas[time_range].append(area_name)

    def add_objs(self, obj_names, period):
        """
        Add object names for a period, avoiding duplicates.
        :param obj_names: List of object names
        :param period: Period name
        """
        if period in self.periods_infos:
            # Filter out objects already present in this period.
            new_obj_names = [obj for obj in obj_names if obj not in self.periods_infos[period]["objects"]]

            # Only add new object names.
            if new_obj_names:
                self.periods_infos[period]["objects"].extend(new_obj_names)
                self.periods_to_obj_names[period].extend(new_obj_names)
                self.obj_names.update(new_obj_names)


    def add_activity(self, activity_name, period):
        """
        Add an activity label for a period.
        :param activity_name: Activity name
        :param period: Period name
        """
        # Validate that period matches the expected time format.
        time_pattern = r"^\d{2}:\d{2}:\d{2}(-\d{2}:\d{2}:\d{2})?$"

        # Check whether period matches the required format.
        if not re.match(time_pattern, period):
            print(f"format error: {period} is not in the correct format")
            return
        if period in self.periods_infos:
            self.periods_infos[period]["activities"].append(activity_name)
            self.periods_to_activities[period] = activity_name
        else:
            # Try to find an existing period that contains this time range.
            matched_periods = spilit_video.find_time_range(period, self.periods_infos.keys())
            if matched_periods:
                for matched_period in matched_periods:
                    self.periods_infos[matched_period]["activities"].append(activity_name)
                    self.periods_to_activities[matched_period] = activity_name

    def output_periods_info(self):
        """
        Format a readable summary of periods.
        :return: Summary string for each period
        """
        lines = []
        for period, info in self.periods_infos.items():
            desc = info.get("description", "")
            areas = ", ".join(info.get("areas", [])) or ""
            objects = ", ".join(info.get("objects", [])) or ""
            activities = "\n".join(info.get("activities", [])) or ""

            lines.append(f"Time period {period}:")
            lines.append(f"  Description: {desc}")
            lines.append(f"  Areas: {areas}")
            lines.append(f"  Objects: {objects}")
            # lines.append(f"  Activities: {activities}")
        return "\n".join(lines)

    def output_periods_description(self):
        """
        Format a readable summary of periods (description-only).
        :return: Summary string for each period
        """
        lines = []
        for period, info in self.periods_infos.items():
            desc = info.get("description", "")
            areas = ", ".join(info.get("areas", [])) or ""
            objects = ", ".join(info.get("objects", [])) or ""

            lines.append(f"Time period {period}:")
            lines.append(f"  Description: {desc}")
            lines.append(f"  Areas: {areas}")
            lines.append(f"  Objects: {objects}")
        return "\n".join(lines)

    def get_entities_in_period(self, period):
        """
        Get all entities within a period: persons, areas, and objects.

        :param period: Period in "hh:mm:ss-hh:mm:ss" format
        :return: Dict of entities grouped by type
        """
        # Initialize result.
        entities = {
            "persons": [],
            "areas": [],
            "objects": []
        }

        # If period does not exist in the navigation graph, return empty.
        if period not in self.periods_infos:
            return entities

        # Areas.
        if "areas" in self.periods_infos[period]:
            entities["areas"] = self.periods_infos[period]["areas"]

        # Objects.
        if "objects" in self.periods_infos[period]:
            entities["objects"] = self.periods_infos[period]["objects"]

        # Persons typically span the whole video; include all known persons.
        entities["persons"] = list(self.person_names)

        return entities



