# Data Directory

CSV files are **not stored in this repository** (`data/*.csv` is gitignored).

Place the 8 dunnhumby Complete Journey CSV files in an external directory and set `DATA_DIR` in your `.env` file:

```env
DATA_DIR=../shopper-data
```

Required files:

- `transaction_data.csv`
- `product.csv`
- `hh_demographic.csv`
- `campaign_table.csv`
- `campaign_desc.csv`
- `coupon.csv`
- `coupon_redempt.csv`
- `causal_data.csv`

Default external location: `../shopper-data/` (sibling to the repo root).
