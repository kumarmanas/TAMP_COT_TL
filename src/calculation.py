from statistics import mode

def fill_with_none(d, n):
    return {k: v + ["None" + str(i) for i in range(n - len(v))] for k, v in d.items()}

def most_freq(l):
    if len(l) == 0:
        return "No output"
    try:
        return mode(l)
    except:
        return l[0]
    
def get_overlaps(explain_dict_list, k):
    return [d[k] for d in explain_dict_list if k in d]

def merge_dicts(explain_dict_list):
    merge_dict = {}
    for d in explain_dict_list:
        for k, v in d.items():
            merge_dict.setdefault(k.lower(), []).append(v)
    return merge_dict

def all_equal(l):
    return all(x == l[0] for x in l)

def count_occurences(l, el):
    return l.count(el)

def calc_certainty_score(l, el, n):
    occ = count_occurences(l, el)
    return occ / n

def ambiguity_detection_ast():
    return

def add_certainty_and_reduce(merge_d, n):
    reduced = {}
    for k, v in merge_d.items():
        certainty_list = []
        for e in v:
            score = calc_certainty_score(v, e, n)
            if not e.startswith("None"):
                certainty_list.append((e, round(score * 100, 2)))
        certainty_list = list(set(certainty_list))
        certainty_list = sorted(certainty_list, key=lambda x: x[1], reverse=True)
        reduced[k] = certainty_list
    return reduced

def ambiguity_detection_translations(explain_dict_list, n, locked_translations):
    merge_d = merge_dicts(explain_dict_list)
    merge_d = fill_with_none(merge_d, n)
    reduced_d = add_certainty_and_reduce(merge_d, n)
    reduced_d = add_locked_subtranslation(reduced_d, locked_translations)
    certainty_triple_list = [
        (
            k,
            [e[0] for e in reduced_d[k]],
            [e[1] for e in reduced_d[k]],
            [e[2] for e in reduced_d[k]],
        )
        for k in reduced_d.keys()
    ]
    return sorted(certainty_triple_list, key=lambda x: max(x[2]))

def add_locked_subtranslation(model_subt, locked_subt):
    model_subt = {k: [(e[0], e[1], False) for e in model_subt[k]] for k in model_subt}
    for k in locked_subt:
        if k in model_subt:
            elem = None
            for e in model_subt[k]:
                if e[0] == locked_subt[k]:
                    elem = e
            if elem is None:
                model_subt[k] = [(locked_subt[k], 0.0, True)] + model_subt[k]
            else:
                model_subt[k].remove(elem)
                model_subt[k] = [(locked_subt[k], elem[1], True)] + model_subt[k]
        else:
            model_subt[k] = [(locked_subt[k], 0.0, True)]
    return model_subt

def ambiguity_final_translation(parsed_result_formulas, n):
    mf = most_freq(parsed_result_formulas)
    cert = calc_certainty_score(parsed_result_formulas, mf, n)
    return (mf, cert)

def transform_translations(translation_data):
    """
    Transform translation data into separate arrays for processing.
    
    Args:
        translation_data: List of tuples containing translation information
        
    Returns:
        Structured arrays with separated translation components
    """
    # Initialize result containers
    result = {
        "keys": [],
        "values": [],
        "confidence": [],
        "status": []
    }
    
    # Process each translation entry
    for entry in translation_data:
        if not entry or len(entry) < 4:
            continue
            
        # Extract components using indices
        key = entry[0]
        value_list = entry[1]
        confidence_list = entry[2]
        status_list = entry[3]
        
        # Add to result containers
        result["keys"].append(key)
        result["values"].append(value_list)
        result["confidence"].append(confidence_list)
        result["status"].append(status_list)
    
    # Return as a list for backward compatibility
    return [
        result["keys"],
        result["values"], 
        result["confidence"],
        result["status"]
    ]

# To maintain the old function name as an alias for backward compatibility
def generate_intermediate_output(intermediate_translation):
    """Legacy alias for transform_translations"""
    return transform_translations(intermediate_translation)
