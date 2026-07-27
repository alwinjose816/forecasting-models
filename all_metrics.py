from forecast_deep_demand_transitions import (
    run_forecast_deep_demand_transitions
)

import pandas as pd

result = run_forecast_deep_demand_transitions(
    "Overall"
)

df = pd.DataFrame(
    result["regimes"]
)

print(df.columns.tolist())