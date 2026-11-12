# Data

The raw data are **not** redistributed with this repository. They are publicly available from the Kenneth R. French Data Library and are fetched by `scripts/download_data.py`.

## Files used

| File in `data/` | Panel | Archive | French section start | SHA-256 of archive |
|---|---|---|---|---|
| `100_Portfolios_10x10_Daily.csv` | Size–book-to-market (main) | `100_Portfolios_10x10_Daily_CSV.zip` | line 19 (value-weighted block) | `8fad24e0…5a0775` |
| `100_Portfolios_ME_OP_10x10_Daily.csv` | Size–operating profitability | `100_Portfolios_ME_OP_10x10_Daily_CSV.zip` | line 23 | `af2c9502…c164af6` |
| `100_Portfolios_ME_INV_10x10_daily.csv` | Size–investment | `100_Portfolios_ME_INV_10x10_Daily_CSV.zip` | line 17 | `85f88811…faf67bf` |

Source: <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html> (daily 10×10 portfolios formed on size and book-to-market, size and operating profitability, size and investment; value-weighted).

## Conventions used by the reader

- Missing codes `-99.99` and `-999` are converted to `NaN`.
- Only the designated value-weighted block is read; the number of header lines before each block is panel-specific and is set by the `section` argument of `data_pipeline.read_section`.
- The main panel covers 6,726 trading days (2000-01-03 to 2026-09-30); after dropping all-empty rows the missing rate is about 0.36%.
- The common evaluation window is 2015-01 to 2026-09 (2,953 forecast dates).
- The three panels overlap in constituents and must not be treated as 300 independent series.

## Checksums

Archives are verified on download. If the French library revises a file, the checksum check fails with a warning rather than silently using different data; note the revision in any replication report.
