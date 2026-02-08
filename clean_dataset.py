import pandas as pd

# Load dataset
df = pd.read_csv('dataset/IoT_dataset.csv')
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── 1. Clean Device ID: D00001 → 1, D00002 → 2, etc. ──
df['Device ID'] = df['Device ID'].str.replace('D', '', regex=False).astype(int)

# ── 2. Normalize Microcontroller duplicates ──
micro_rename = {
    'Nordic nRF52832': 'nRF52832',
    'Silicon Labs EFR32MG21': 'EFR32MG21',
    'Raspberry Pi BCM2837': 'Raspberry Pi (BCM2837)',
    'ESP32 CAM': 'ESP32',          # ESP32 CAM is an ESP32 variant
    'ESP32-C3': 'ESP32',           # ESP32-C3 is an ESP32 variant
}
df['Microcontroller'] = df['Microcontroller'].replace(micro_rename)

# Verify we now have 11 unique microcontrollers
unique_micros = sorted(df['Microcontroller'].unique())
print(f"\nNormalized Microcontrollers ({len(unique_micros)}):")
for i, m in enumerate(unique_micros, 1):
    print(f"  {i} → {m}")

# ── 3. Encode Device Type (30 types → 1-30) ──
device_types = sorted(df['Device Type'].unique())
device_type_map = {name: idx for idx, name in enumerate(device_types, 1)}
print(f"\nDevice Type Mapping ({len(device_type_map)}):")
for name, idx in device_type_map.items():
    print(f"  {idx} → {name}")

df['Device Type'] = df['Device Type'].map(device_type_map)

# ── 4. Encode Microcontroller (11 types → 1-11) ──
micro_map = {name: idx for idx, name in enumerate(unique_micros, 1)}
print(f"\nMicrocontroller Mapping ({len(micro_map)}):")
for name, idx in micro_map.items():
    print(f"  {idx} → {name}")

df['Microcontroller'] = df['Microcontroller'].map(micro_map)

# ── 6. Save cleaned dataset ──
df.to_csv('dataset/IoT_dataset_cleaned.csv', index=False)

print(f"\nCleaned shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nData types:")
print(df.dtypes)
print(f"\nSaved to dataset/IoT_dataset_cleaned.csv")
