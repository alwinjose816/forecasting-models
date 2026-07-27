from load_data import load_dealer_orders
from demand_memory import run_demand_memory

import pandas as pd
def classify_memory(emh):

    if emh < 60:
        return "Short"

    elif emh < 120:
        return "Medium"

    elif emh < 180:
        return "Long"

    return "Extended"


def analyze_all_products():

    df = load_dealer_orders()

    products = sorted(
        df["product_code"].unique()
    )
   

    results = []

    # Overall first
    overall_memory = run_demand_memory("Overall")

    results.append({

        "product": "Overall",

        "ms": round(
            overall_memory["ms"], 4
        ),

        "weekly_memory": round(
            overall_memory["weekly_memory"], 4
        ),

        "monthly_memory": round(
            overall_memory["monthly_memory"], 4
        ),

        "annual_memory": round(
            overall_memory["annual_memory"], 4
        ),

        "ml50": overall_memory["ml50"],

        "emh": overall_memory["emh"],

        "ml": overall_memory["ml"],

        "mp": round(
            overall_memory["mp"], 4
        ),

        "ss": round(
            overall_memory["ss"], 4
        ),

        "entropy": round(
            overall_memory["entropy"], 4
        ),

        "dmi": round(
            overall_memory["dmi"], 4
        ),

        "memory_type": classify_memory(
            overall_memory["emh"]
        )

    })

    for product in products:

        try:

            memory = run_demand_memory(product)

            results.append({

                "product": product,

                "ms": round(memory["ms"], 4),

                "weekly_memory": round(
                    memory["weekly_memory"], 4
                ),

                "monthly_memory": round(
                    memory["monthly_memory"], 4
                ),

                "annual_memory": round(
                    memory["annual_memory"], 4
                ),

                "ml50": memory["ml50"],

                "emh": memory["emh"],

                "ml": memory["ml"],

                "mp": round(
                    memory["mp"], 4
                ),

                "ss": round(
                    memory["ss"], 4
                ),

                "entropy": round(
                    memory["entropy"], 4
                ),

                "dmi": round(
                    memory["dmi"], 4
                ),

                "memory_type": classify_memory(
                    memory["emh"]
                )

            })

        except Exception as e:

            print(product, e)

    return pd.DataFrame(results)