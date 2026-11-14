"""Data stage: download the HHS facility file, load it into Parquet, clean and validate it.

download  ->  data/raw/<csv> + SHA256SUMS      (the file exactly as HHS serves it)
ingest    ->  data/interim/raw.parquet          (all 128 columns, typed, nothing removed)
clean     ->  data/clean/beds_weekly.parquet    (hospital x week grid the model reads)
              data/clean/hospitals.parquet      (one row per hospital, with its split)
              data/clean/counts.json            (every count we report, regenerated)
"""
