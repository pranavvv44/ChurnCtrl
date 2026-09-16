import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(
    f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASS")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'
)

df = pd.read_sql('SELECT * FROM customers_bank', engine)
df.to_csv('customers_bank_export.csv', index=False)
print(f'Exported {len(df)} rows')
