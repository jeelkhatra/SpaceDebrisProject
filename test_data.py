from modules.data_engine import fetch_orbital_data

df = fetch_orbital_data()

print("\n===== SATELLITE DATA =====")
print(df.head())

print("\nNumber of objects:", len(df))

print("\n===== COLUMNS =====")
print(df.columns.tolist())
