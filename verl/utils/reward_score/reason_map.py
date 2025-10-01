import re
import json
import numpy as np
import logging
import sys
import os
from typing import Optional
from verl.utils.reward_score.reasonmap_utils import find_math_answer


def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("reasonmap")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        f = open(log_file, mode="w", buffering=1, encoding="utf-8")
        fh = logging.StreamHandler(f)  
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
logger = setup_logger(log_file="./reward.txt")




def clean_string(s):
    s = s.replace('Metro', '').replace('metro', '').replace(',', '').replace('*', '').replace('\n', '').replace("'", '').replace('Univ.', 'University').replace('Univ', 'University').replace('Tech.', 'Technology').replace('Tech', 'Technology').replace('road', 'lu').replace('Road', 'lu')
    s = s.replace('Station', '').replace('station', '').replace('Asian Games Village', 'yayuncun')
    if s.endswith('站'):
        s = s[:-1]
    s = re.sub(r'[\(\（][^\)\）]*[\)\）]', '', s)
    return re.sub(r'[\s\-]+', '', s).lower()


def get_acc_torf_counting(type_reasonmap, model_answer, ground_truth):
    if "TorF" in type_reasonmap:
        ground_truth = "yes" if ground_truth == 1 else "no"
        acc = model_answer.lower() == ground_truth
    elif type_reasonmap == "Counting2" or type_reasonmap == "Counting3":
        try:
            model_answer = int(model_answer)
            acc = int(model_answer) == ground_truth
        except (ValueError, TypeError):
            acc = False
    elif type_reasonmap == "Counting1":
        model_answer = model_answer.lower()

        if model_answer in ['a', 'b', 'c', 'd']:
            mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            model_answer = int(mapping[model_answer])
            acc = model_answer == ground_truth
        else:   
            acc = False
    return acc


def get_acc_planning(model_answer, metro_data, station1, station2):
    acc = False
    route_sections = model_answer.split("--")
    route_data = []

    for section in route_sections:
        # section_lc = section.lower()
        if section.strip():
            if "Route Name:" not in section or "Departure Stop:" not in section or "Arrival Stop:" not in section:
                print("Invalid section format. Skipping...")
                continue
            route_info = {}
            route_name_match = re.search(r"Route Name: (.*?)\n", section)
            departure_match = re.search(r"Departure Stop: (.*?)\n", section)

            route_info['Route Name'] = route_name_match.group(1).strip() if route_name_match else "Wrong"
            route_info['Departure Stop'] = clean_string(departure_match.group(1)) if departure_match else "Wrong"
            # route_info['Route Name'] = re.search(r"Route Name: (.*?)\n", section).group(1).strip()
            # route_info['Departure Stop'] = clean_string(re.search(r"Departure Stop: (.*?)\n", section).group(1))

            route_info['Arrival Stop'] = clean_string(section.split("Arrival Stop:")[-1].strip())


            route_data.append(route_info)

    logger.info(f"route_data:{route_data}")

    # if can not get route data, pass this sample, we assume the answer format is wrong
    if len(route_data) == 0:
        print("No route data found because of wrong format.")
    else:
        first_route = route_data[0] # first route section
        last_route = route_data[-1] # last route section

        route_set = set()

        # Check if the first route's Departure Stop matches station1 and the last route's Arrival Stop matches station2
        if clean_string(first_route['Departure Stop']) == clean_string(station1) and clean_string(last_route['Arrival Stop']) == clean_string(station2):

            correct_route = True
            for i in range(len(route_data)):
                route = route_data[i]
                route_name = route['Route Name']

                if "八通" in route_name:
                    route_name = "Line 1"
                if "大兴" in route_name:
                    route_name = "Line 4"
                departure_stop = route['Departure Stop']
                arrival_stop = route['Arrival Stop']

                if route_name in metro_data:
                    stations = [clean_string(station) for station in metro_data[route_name]]

                    # Check if departure_stop and arrival_stop are in the correct order
                    if clean_string(departure_stop) in stations and clean_string(arrival_stop) in stations:
                        # For transfer check, if it's not the last route, the Arrival Stop of the current route
                        # should match the Departure Stop of the next route
                        if i < len(route_data) - 1:
                            next_route = route_data[i + 1]
                            if clean_string(arrival_stop) != clean_string(next_route['Departure Stop']):
                                correct_route = False
                                print(f"Route {route_name}: Incorrect transfer from {arrival_stop} to {next_route['Departure Stop']}.")
                    else:
                        correct_route = False
                        print(f"Route {route_name}: One or both stops not found in route.")
                else:
                    correct_route = False
                    print(f"Route {route_name}: Route name not found in metro data.")

            if correct_route:
                acc = True
                print("The route is correct.")
            else:
                print("The route is incorrect.")
        else:
            print(f"Wrong departure or arrival station. Expected {station1} and {station2}, but got {first_route['Departure Stop']} and {last_route['Arrival Stop']}.")

    return acc


def get_difficulty_weight(type_reasonmap, question_transfer_count, difficulty_city):
    weight = 1.0

    def difficulty_city_weight(difficulty_city):
        if difficulty_city == "easy":
            return 1.0
        elif difficulty_city == "middle":
            return 1.2
        elif difficulty_city == "hard":
            return 1.5
        else:
            raise NotImplementedError(f"Difficulty city {difficulty_city} is not recognized.")

    if "TorF" in type_reasonmap or "Counting" in type_reasonmap:
        weight = difficulty_city_weight(difficulty_city)
    elif "planning" in type_reasonmap:
        if question_transfer_count == 0:
            weight = difficulty_city_weight(difficulty_city)
        elif question_transfer_count > 0:
            weight = difficulty_city_weight(difficulty_city) + 0.5
        else:
            raise NotImplementedError(f"Question transfer count {question_transfer_count} is not valid.")
    else:
        raise NotImplementedError(f"Type {type_reasonmap} is not recognized.")
    
    return weight


def get_detailed_reward_counting1(stops_answer_counting1, metro_data, station1, station2):
    from typing import List, Dict
    def norm(s: str) -> str:
        return s.strip().lower()

    logger.info(f"stops_answer_counting1: {stops_answer_counting1}")
    s1, s2 = clean_string(station1), clean_string(station2)
    
    # print(f"s1: {s1}, s2: {s2}")
    # print(f"metro_data: {metro_data}")

    line_stations = None
    for stations in metro_data.values():
        names_norm = [clean_string(x) for x in stations]
        if s1 in names_norm and s2 in names_norm:
            line_stations = stations
            break
    if line_stations is None:
        raise ValueError("station1 and station2 are not on the same line.")

    names_norm = [clean_string(x) for x in line_stations]
    i1, i2 = names_norm.index(s1), names_norm.index(s2)
    lo, hi = (i1, i2) if i1 < i2 else (i2, i1)
    gt_norm_set = set(names_norm[lo + 1 : hi])

    reward = 0.0
    seen_correct = set()
    for pred in stops_answer_counting1:
        p = clean_string(pred)
        if p in gt_norm_set:
            if p not in seen_correct:
                reward += 0.5
                logger.info(f"pred: {p}, reward += 0.5")
                seen_correct.add(p)
            else:
                reward -= 0.5
                logger.info(f"pred: {p}, reward -= 0.5, repeat")
        else:
            reward -= 0.5 # wrong stop penalty
            logger.info(f"pred: {p}, reward -= 0.5, wrong stop")

    return reward


def get_detailed_reward_planning(model_answer, metro_data, station1, station2, question_transfer_count):
    route_sections = model_answer.split("--")
    route_data = []

    score = 0.0

    for section in route_sections:
        section_lc = section.lower()
        if section.strip():
            if "route name:" not in section_lc or "departure stop:" not in section_lc or "arrival stop:" not in section_lc:
                print("Invalid section format. Skipping...")
                continue
            route_info = {}
            route_name_match = re.search(r"Route Name: (.*?)\n", section)
            departure_match = re.search(r"Departure Stop: (.*?)\n", section)

            route_info['Route Name'] = route_name_match.group(1).strip() if route_name_match else "Wrong"
            route_info['Departure Stop'] = clean_string(departure_match.group(1)) if departure_match else "Wrong"

            route_info['Arrival Stop'] = clean_string(section.split("Arrival Stop:")[-1].strip())


            route_data.append(route_info)

    # if can not get route data, pass this sample, we assume the answer format is wrong (give a punishment)
    if len(route_data) == 0:
        print("No route data found because of wrong format.")
    else:
        first_route = route_data[0] # first route section
        last_route = route_data[-1] # last route section

        route_set = set()

        # Check if the first route's Departure Stop matches station1 and the last route's Arrival Stop matches station2
        if clean_string(first_route['Departure Stop']) == clean_string(station1) and clean_string(last_route['Arrival Stop']) == clean_string(station2):
            # Verify each route and ensure transfer points match
            score += 2

            correct_route = True
            for i in range(len(route_data)):

                if (i+1) > question_transfer_count:
                    score -= 5

                route = route_data[i]
                route_name = route['Route Name']

                if "八通" in route_name:
                    route_name = "Line 1"
                if "大兴" in route_name:
                    route_name = "Line 4"
                departure_stop = route['Departure Stop']
                arrival_stop = route['Arrival Stop']

                if route_name in metro_data:
                    if question_transfer_count == 0:
                        for route, stations in metro_data.items():
                            if clean_string(station1) in [clean_string(s) for s in stations] and clean_string(station2) in [clean_string(s) for s in stations]:
                                if route == route_name:
                                    score += 4
                                    break
                    else:
                        score += 3
                    stations = [clean_string(station) for station in metro_data[route_name]]

                    # Check if departure_stop and arrival_stop are in the correct order
                    if clean_string(departure_stop) in stations and clean_string(arrival_stop) in stations:
                        # For transfer check, if it's not the last route, the Arrival Stop of the current route
                        # should match the Departure Stop of the next route
                        if i < len(route_data) - 1:
                            next_route = route_data[i + 1]
                            if clean_string(arrival_stop) != clean_string(next_route['Departure Stop']):
                                correct_route = False
                                print(f"Route {route_name}: Incorrect transfer from {arrival_stop} to {next_route['Departure Stop']}.")
                            else:
                                if (i+1) > question_transfer_count:
                                    pass
                                else:
                                    score += 1
                    else:
                        correct_route = False
                        print(f"Route {route_name}: One or both stops not found in route.")
                else:
                    correct_route = False
                    print(f"Route {route_name}: Route name not found in metro data.")

            if correct_route:
                print("The route is correct.")
            else:
                print("The route is incorrect.")
        else:
            print(f"Wrong departure or arrival station. Expected {station1} and {station2}, but got {first_route['Departure Stop']} and {last_route['Arrival Stop']}.")

    score = min(score, 10.0)

    return score


def reason_map_compute_score(response, ground_truth, station1, station2, metro_data, mode, type_reasonmap, difficulty_city, question_transfer_count, **kwargs) -> float:
    score = 0.0
    acc = None
    logger.info(f"type: {type_reasonmap}")

    # @sicheng: get answer by parsing 
    model_answer = None # str
    stops_answer_counting1 = []
    if "TorF" in type_reasonmap or "Counting" in type_reasonmap:
        model_answer = response
        if 'oxed{' not in model_answer:
                for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
                    flag = flag.replace('the', 'The')
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
        elif model_answer.count('oxed{') > 1:
            model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]

        model_answer = find_math_answer(model_answer).rstrip('.').lstrip(':').strip()

        # @sicheng: add for Counting1 question
        if type_reasonmap == "Counting1":
            from verl.utils.reward_score.reasonmap_utils import extract_stops_no_quotes
            stops_answer_counting1 = extract_stops_no_quotes(response)
        
    elif "planning" in type_reasonmap:
        model_answer = response
    else:
        raise NotImplementedError(f"Type {type_reasonmap} should not show!")
    
    # @sicheng: get acc
    if "TorF" in type_reasonmap or "Counting" in type_reasonmap:
        acc = get_acc_torf_counting(type_reasonmap, model_answer, ground_truth)
    elif "planning" in type_reasonmap:
        acc = get_acc_planning(model_answer, metro_data, station1, station2)
    else:
        raise NotImplementedError(f"Type {type_reasonmap} should not show!")

    # @sicheng: get score for accuracy & format
    if acc:
        score += 10

    # @sicheng: get detailed score
    extra_score_weighting = 0.5
    if type_reasonmap == "Counting1":
        pass
    elif type_reasonmap == "planning":
        score += extra_score_weighting * get_detailed_reward_planning(model_answer, metro_data, station1, station2, question_transfer_count)
        pass

    # @sicheng: get weight
    weight = 1.0
    if mode == "baseline":
       pass
    elif mode == "difficulty_aware":
        weight = get_difficulty_weight(type_reasonmap, question_transfer_count, difficulty_city)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented for ReasonMap compute score.")
    
    score *= weight
    
    logger.info(f"acc: {acc}")
    logger.info(f"score: {score}")
    logger.info("-"*20)
    return score
