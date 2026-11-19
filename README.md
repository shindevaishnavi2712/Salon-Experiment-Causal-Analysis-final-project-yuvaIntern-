# Salon Experiment & Causal Analysis

A reproducible Python project that evaluates a fictional salon-chain business initiative using:
- A/B test design and sample-size / power calculations
- Randomization balance and Sample Ratio Mismatch (SRM) checks
- Difference-in-means testing with confidence intervals
- Bootstrap confidence intervals
- Difference-in-Differences (DiD) for a historical rollout without a randomized experiment
- CUPED-style variance reduction and regression adjustment
- Outlet and customer-segment analysis
- Visualizations and a concise decision recommendation

## Important data note
No real salon dataset was supplied with the assignment. This project therefore uses **synthetic demonstration data** representing 22 fictional outlets. Do not describe the generated records as real customer data or claim that the estimated effects are verified business results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py --rows 300000 --output data/salon_appointments.csv
python src/run_analysis.py --input data/salon_appointments.csv --output outputs
```

To run a smaller quick demonstration:
```bash
python src/generate_data.py --rows 20000 --output data/salon_appointments.csv
python src/run_analysis.py --input data/salon_appointments.csv --output outputs
```

## Dataset fields
- `appointment_id`: unique appointment record
- `outlet_id`: one of 22 fictional outlets
- `appointment_date`: appointment date
- `customer_id`: synthetic customer identifier; walk-ins may be missing
- `service_category`: service category
- `stylist_id`: synthetic stylist identifier
- `channel`: booking channel
- `group`: A/B assignment (`control` or `treatment`)
- `period`: `pre` or `post` period for historical DiD illustration
- `revenue`: appointment revenue in INR
- `discount_pct`: discount percentage
- `no_show`: whether the customer did not attend
- `repeat_within_60d`: synthetic indicator for repeat visit within 60 days
- `prior_revenue`: pre-period customer revenue proxy used for CUPED-style adjustment

## Outputs
The analysis writes:
- `summary_metrics.csv`
- `ab_test_results.csv`
- `did_results.csv`
- `segment_results.csv`
- `fig_revenue_by_group.png`
- `fig_no_show_by_group.png`
- `fig_did_trends.png`
- `fig_feature_importance.png`

## Interpretation
The script reports estimated effects and 95% confidence intervals. A statistically significant result is not automatically a worthwhile business result: consider effect size, implementation cost, customer experience, and guardrail metrics. The DiD estimate relies on the parallel-trends assumption, which cannot be proven from the post-period comparison alone.
