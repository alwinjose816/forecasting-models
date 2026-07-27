from supabase import create_client
import pandas as pd

SUPABASE_URL = "https://yvuxjdpvvtpbngoubqgq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl2dXhqZHB2dnRwYm5nb3VicWdxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg1OTY1MTEsImV4cCI6MjA5NDE3MjUxMX0.kI3mBen9jhQ4avBkL93xkAZg27dVvyE7NJACPECrKsE"

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

def load_dealer_orders():

    response = (
        supabase
        .table("dealer_orders")
        .select(
            "order_date,product_code,total_weight_mt"
        )
        .execute()
    )

    df = pd.DataFrame(response.data)

    df["order_date"] = pd.to_datetime(
        df["order_date"]
    )

    return df