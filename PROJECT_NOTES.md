# Project notes and methodology

## Business question
Did the salon initiative improve revenue per appointment and customer outcomes, beyond ordinary variation and seasonal patterns?

## Hypotheses
- H0: mean revenue per appointment is equal in treatment and control.
- H1: mean revenue per appointment differs between treatment and control.
- Guardrails: no-show rate should not increase materially; customer experience and discount costs should be monitored.

## Experiment design
Randomly assign eligible outlets or customers to control and treatment before launch. Keep assignment stable, define eligibility in advance, log exposure, and avoid changing the primary metric after seeing results. The included generator creates illustrative 50:50 groups.

## Statistical methods
- Welch's independent-samples t-test and 95% confidence interval for mean revenue.
- Two-proportion z-test for no-show rates.
- Chi-square Sample Ratio Mismatch check against a 50:50 allocation.
- Two-sided 5% significance level and 80% power planning; sample-size calculation shown for standardized effect size d=0.20. Replace this planning effect with a business-relevant minimum detectable effect before a real experiment.
- Bootstrap is listed as an extension: resample at the randomization unit (for example outlet or customer), not individual appointments, when assignment occurs at cluster level.
- CUPED-style adjustment uses a pre-treatment covariate to reduce outcome variance. Confirm that the covariate is measured before treatment.
- DiD estimates the group-by-period interaction. Its causal interpretation requires parallel pre-trends, no differential concurrent interventions, stable group composition, and no spillovers.

## Decision framework
Recommend GO only when the primary metric has a positive, commercially meaningful effect, uncertainty is acceptable, and guardrails remain within pre-agreed limits. Recommend NO-GO if harm is credible. Recommend EXTEND if the confidence interval is too wide or data quality / assumptions are insufficient.

## Limitations
The generated dataset is synthetic and intentionally simplified. It is not a real operational dataset, and its simulated effects must not be presented as verified findings. In a real salon dataset, validate definitions of cancellations/no-shows, missing client IDs for walk-ins, service-name mapping across outlets, duplicate appointments, refunds, and revenue recognition.
