from demand_memory import run_demand_memory
import numpy as np


def run_dmsm_forecast(product):

    memory = run_demand_memory(product)

    print(memory.keys())

    y = np.array(memory["demand"])

    memory_state = np.array(
        memory["memory_state"]
    )

    forecast = [None]

    for i in range(1, len(y)):

        forecast.append(
            float(
                memory_state[i - 1]
            )
        )

    actual = y[1:]
    pred = np.array(
        forecast[1:]
    )

    mae = np.mean(
        np.abs(actual - pred)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - pred) ** 2
        )
    )

    smape = np.mean(
        2
        * np.abs(actual - pred)
        /
        (
            np.abs(actual)
            + np.abs(pred)
            + 1e-8
        )
    ) * 100

    return {

        "dates":
        memory["dates"],

        "actual":
        y.tolist(),

        "memory_state":
        memory_state.tolist(),

        "forecast":
        forecast,

        "mae":
        float(mae),

        "rmse":
        float(rmse),

        "smape":
        float(smape),

        "ml50":
        memory["ml50"],

        "memory_type":
        memory["memory_type"]

    }