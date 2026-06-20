# German CEO Executive Compensation Dataset

Quick breakdown of the data pipeline for our German CEO executive compensation project (2006–2021). We've got three files tracking 155 listed German companies.

**TL;DR for the team:** If you just want one file to rule them all, just grab `ceo_year_lens1.csv`—it has every single column from stages 1, 2, and 3. I'm keeping all three files here so you can easily cross-check each stage if anything looks weird.

---

## 📁 File Structure & Pipeline

### 1. `ceo_year_clean.csv` (The Foundation)
* **Specs:** 1,267 rows × 35 cols (~285 KB)
* **What it is:** A clean, CEO-only panel dataset. All the data landmines have been cleared out. It contains one row per CEO per year from 2006 to 2021.
* **When to use it:** If you want to build your own model variants with a different target variable or a totally custom feature set, start here.

#### Key Columns:
* **Identity:** `isin`, `year`, `company_shortname`, `exec_id`, `exec_fullname`
* **Raw Pay (in thousands EUR):** `salary`, `one_year_bonus`, `multi_year_bonus`, `total_equity_grants`, `pension`, `total_comp`
* **Peer Group Keys:** `index_listing` (DAX/MDAX), `n_executives`
* **Bio Data:** `female`, `nationality`, `date_of_birth`
* **Sector:** `sector` (Heads up: only ~250 rows for DAX15 have this, the rest are `NaN`—this is by design!)
* **Derived:** `tenure_years` (calculated from `date_begin_ceo` to the end of the fiscal year)

#### Data Cleaning Done Under the Hood:
* Dropped 29 rows where `total_comp <= 0`.
* Flushed out raw data errors in `pension` by setting them to `NaN` for 55 rows where the value was negative or > EUR 50M (e.g., one crazy entry was sitting at EUR 150M).
* Filtered strictly to `ceo_flag_eoy == 1` (CEOs in office at the end of the fiscal year).

---

### 2. `ceo_year_features.csv` (The Feature Pack)
* **Specs:** 1,267 rows × 48 cols (~491 KB)
* **What it is:** A superset of the clean data, packed with feature engineering and matched with ORBIS financial data.
* **When to use it:** Perfect if you want to build Lens 2 or Lens 3 from scratch without loading the full Lens 1 model outputs, or if you want to visualize pay-mix breakdowns.

#### Added Features vs Clean:
* **Pay-Mix %:** `pct_fixed`, `pct_sti`, `pct_lti`, `pct_other` (Sanity check: the medians are 33/31/30/1%, which perfectly matches our brief).
* **Index Listing Fix:** `index_listing` has been forward-filled (boosting valid rows from 1211 to 1264, achieving 99.8% coverage).
* **ORBIS Financials:** Pulled from OPRE, OPPL, ROA, ROE, TOAS, EMPL, and STAF. Deduped to exactly one row per `isin` × `year`, prioritizing consolidated statements. Includes: `revenue`, `ebit`, `roa`, `roe`, `total_assets`, `employees`, `staff_costs`.
* **Derived Metrics:** `log_revenue`, `ebit_margin`.

#### ⚠️ Important Coverage Note:
ORBIS coverage is around **81% overall**. Banking and Insurance are at **0%** because firms like Allianz, Deutsche Bank, and Munich Re are completely absent from ORBIS (a known data loophole). For Lens 3, you **must** fetch the shareholder return data from the XLSX file for these three specific firms.

---

### 3. `ceo_year_lens1.csv` (The Money File 💰)
* **Specs:** 1,267 rows × 55 cols (~617 KB)
* **What it is:** The superset of the features dataset plus the actual Lens 1 model outputs. This is what the Streamlit dashboard reads directly.
* **When to use it:** Use this as your primary data source for dashboards, pitch screenshots, CEO drill-downs, scatter plots (like residual vs revenue), etc.

#### Added Features vs Features:
* **Model Predictions:** `expected_log_comp`, `expected_comp` (Generated via a Ridge regression model with GroupKFold by ISIN, hitting a CV R² = 0.379).
* **Residuals:** `actual_log_comp`, `residual_log` (Actual minus expected on a log scale).
* **Peer Benchmarking:** `level_robust_z` (Robust z-score using median + MAD of residuals *within* the specific `index_listing` and `year` peer group).
* **Flags:** `level_flag` (`True` if |robust_z| > 3, meaning the CEO is paid way outside what peer data justifies).
* **Human-Readable Explanations:** `level_reason` (Plain-English summaries for flagged rows, e.g., *"Level +5.2σ — paid over peers (actual €23,450,000, expected €4,640,635, +405% vs DAX 2018 peer model)"*).

#### 📊 Quick Sanity Check on Results:
The model flagged **31 rows (2.4%)** as outliers, and they match existing executive comp literature perfectly:
* **Overpaid Outliers (High σ):** Beiersdorf's Heidenreich (+5.2σ), Linde's Angel (+4.4σ), and Deutsche Bank's Ackermann in 2007 (+3.8σ).
* **Underpaid Outliers (Low σ):** Medion's Brachmann (-8.3σ, founder skipped comp), Zalando's founders (-5.2σ), and Commerzbank's Blessing in 2011 (-5.1σ, financial bailout era).

#### 🛠️ Data Integrity Note:
For 3 rows where `index_listing` is still `NaN` (so we can't build a proper peer group), the Lens 1 columns will also show up as `NaN`. This isn't a bug; it's by design. No peers = no benchmark.

---

## 💡 Quick Note for the Team
If you just want to grab a single file and get moving, take **`ceo_year_lens1.csv`**—it has all the columns from the previous steps baked right into it. However, I’ve pushed all three files to the repo so you can easily cross-check the data at each stage if anything looks off.
