from demand_regimes import run_demand_regimes

import pandas as pd
import numpy as np
MIN_TRANSITION_PROB = 0.10



def run_demand_transitions(product="Overall"):

    # ==========================================
    # LOAD REGIMES
    # ==========================================

    result = run_demand_regimes(product)

    if "error" in result:
        return result

    df = pd.DataFrame(
        result["regimes"]
    )
    # ==========================================
    # STATE NAMES
    # ==========================================

    state_names = result[
        "state_names"
    ]
    if "state" not in df.columns:
        return {
            "error":
            "No states available"
        }
    state_frequency = (
        df["state"]
        .value_counts(normalize=True)
        .sort_index()
        .to_dict()
    )

    # ==========================================
    # SORT BY TIME
    # ==========================================

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = (
            df
            .dropna(subset=["date"])
            .sort_values("date")
        )
    # ==========================================
    # CURRENT -> NEXT STATE
    # ==========================================

    transitions = pd.DataFrame({

        "from_state":
        df["state"],

        "to_state":
        df["state"].shift(-1)

    })

    transitions = (
        transitions
        .dropna()
    )

    transitions["to_state"] = (
        transitions["to_state"]
        .astype(int)
    )

    # ==========================================
    # TRANSITION COUNTS
    # ==========================================

    transition_counts = pd.crosstab(
        transitions["from_state"],
        transitions["to_state"]
    )
    transition_counts.columns = [
        str(c)
        for c in transition_counts.columns
    ]

    # ==========================================
    # TRANSITION PROBABILITIES
    # ==========================================

    transition_matrix = (
        transition_counts
        .div(
            transition_counts.sum(axis=1),
            axis=0
        )
    )
    transition_matrix.columns = [
        str(c)
        for c in transition_matrix.columns
    ]

    # ==========================================
    # STABILITY
    # ==========================================

    # STABILITY

    stability = {}

    for state in transition_matrix.index:

        p = (
            transition_matrix
            .loc[state]
            .get(str(state), 0)
        )

        stability[int(state)] = float(p)

    # ==========================================
    # EXPECTED DURATION
    # ==========================================

    # EXPECTED DURATION

    expected_duration = {}

    for state in transition_matrix.index:

        p = (
            transition_matrix
            .loc[state]
            .get(str(state), 0)
        )

        expected_duration[int(state)] = round(
            (
                1 / (1 - p)
                if p < 0.999
                else np.inf
            ),
            2
        )

  

    # ==========================================
    # MOST LIKELY NEXT STATE
    # ==========================================

    next_state = {}

    for state in transition_matrix.index:

        next_state[int(state)] = int(
            transition_matrix
            .loc[state]
            .idxmax()
        )
   # ==========================================
    # MOST LIKELY CHANGE (EXCLUDING SELF)
    # ==========================================

    most_likely_change = {}

    for state in transition_matrix.index:

        row = (
            transition_matrix
            .loc[state]
            .copy()
        )

        if state in row.index:
            row[state] = 0

        if row.sum() > 0:

            most_likely_change[
                int(state)
            ] = int(
                row.idxmax()
            )

        else:

            most_likely_change[
                int(state)
            ] = None


    # ==========================================
    # CHANGE NAMES
    # ==========================================

    most_likely_change_names = {}

    for state, next_id in (
        most_likely_change.items()
    ):

        current_name = state_names.get(
            state,
            f"State {state}"
        )

        if next_id is None:

            next_name = (
                "No Significant Change"
            )

        else:

            next_name = state_names.get(
                next_id,
                f"State {next_id}"
            )

        most_likely_change_names[
            f"{state}:{current_name}"
        ] = next_name
    next_state_names = {}

    for state in transition_matrix.index:

        next_id = int(
            transition_matrix
            .loc[state]
            .idxmax()
        )

        current_name = state_names.get(
            int(state),
            f"State {state}"
        )

        next_name = state_names.get(
            next_id,
            f"State {next_id}"
        )

        next_state_names[
            f"{int(state)}:{current_name}"
        ] = next_name
    
    transition_labels = []

    for i in transition_matrix.index:

        for j in transition_matrix.columns:

            p = transition_matrix.loc[
                i,
                j
            ]
            

            if p >= MIN_TRANSITION_PROB:
                

                transition_labels.append({

                    "from":
                    state_names.get(
                        int(i),
                        f"State {i}"
                    ),

                    "to":
                    state_names.get(
                        int(j),
                        f"State {j}"
                    ),

                    "probability":
                    float(p),
                    "probability_pct": round(
                        p * 100,
                        2
                    )

                })
    transition_labels = sorted(
        transition_labels,
        key=lambda x: x["probability"],
        reverse=True
    )
    # ==========================================
    # TRANSITION GRAPH EDGES
    # ==========================================

    transition_edges = []

    for item in transition_labels:

        transition_edges.append({

            "source":
            item["from"],

            "target":
            item["to"],

            "weight":
            item["probability"],

            "weight_pct":
            item["probability_pct"],
            "type": (
                "self"
                if item["from"] == item["to"]
                else "transition"
            )

        })
    import json

    response = {

        "transition_counts":
        transition_counts
        .reset_index()
        .to_dict("records"),

        "transition_matrix":
        transition_matrix
        .reset_index()
        .round(4)
        .to_dict("records"),

        "stability":
        {str(k): float(v)
        for k, v in stability.items()},

        "next_state":
        {str(k): int(v)
        for k, v in next_state.items()},

        "state_names":
        {str(k): str(v)
        for k, v in state_names.items()},

        "next_state_names":
        next_state_names,

        "expected_duration":
        {str(k): float(v)
        for k, v in expected_duration.items()},

        "transition_labels":
        transition_labels,

        "state_frequency":
        {str(k): float(v)
        for k, v in state_frequency.items()},

        "most_likely_change":
        {
            str(k): (
                None if v is None else int(v)
            )
            for k, v in most_likely_change.items()
        },

        "most_likely_change_names":
        most_likely_change_names,

        "transition_edges":
        transition_edges
    }

    json.dumps(response)

    return response

    

    # ==========================================
    # RETURN
    # ==========================================

    return {

        "transition_counts":
        transition_counts
        .reset_index()
        .to_dict("records"),

        "transition_matrix":
        transition_matrix
        .reset_index()
        .round(4)
        .to_dict("records"),

        "stability":
        {str(k): float(v)
        for k, v in stability.items()},

        "next_state":
        {str(k): int(v)
        for k, v in next_state.items()},

        "state_names":
        {str(k): str(v)
        for k, v in state_names.items()},

        "next_state_names":
        next_state_names,

        "expected_duration":
        {str(k): float(v)
        for k, v in expected_duration.items()},

        "transition_labels":
        transition_labels,

        "state_frequency":
        {str(k): float(v)
        for k, v in state_frequency.items()},

        "most_likely_change":
        {
            str(k):
            (None if v is None else int(v))
            for k, v in most_likely_change.items()
        },

        "most_likely_change_names":
        most_likely_change_names,

        "transition_edges":
        transition_edges
    }

if __name__ == "__main__":

    result = run_demand_transitions(
        "Overall"
    )

    print(
        "\nTransition Matrix:"
    )

    print(
        pd.DataFrame(
            result[
                "transition_matrix"
            ]
        )
    )

    print(
        "\nState Stability:"
    )

    print(
        result["stability"]
    )

    print(
        "\nMost Likely Next State:"
    )

    print(
        result["next_state"]
    )
    print(
    "\nExpected Duration:"
    )

    print(
        result["expected_duration"]
    )

    print(
        "\nNext State Names:"
    )

    print(
        result["next_state_names"]
    )
    print(
        "\nRegime Evolution:"
    )

    for k, v in result[
        "next_state_names"
    ].items():

        print(
            f"{k} -> {v}"
        )

    print(
        "\nTransition Labels:"
    )

    for item in result["transition_labels"]:
        print(item)
    print(
        "\nState Frequency:"
    )

    for k, v in result[
        "state_frequency"
    ].items():

        print(
            f"State {k}: {v:.2%}"
        )
    print(
        "\nRegime Change Paths:"
    )

    for k, v in result[
        "most_likely_change_names"
    ].items():

        print(
            f"{k} -> {v}"
        )
    print(
        "\nTransition Network:"
    )

    for edge in result[
        "transition_edges"
    ]:

        print(
            f"{edge['source']} -> "
            f"{edge['target']} "
            f"({edge['weight_pct']:.2f}%)"
        )