"""Run A/B test, DiD, CUPED-style adjustment, segments and charts."""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def mean_ci(x, alpha=.05):
    x = pd.Series(x).dropna()
    if len(x) < 2: return np.nan, np.nan, np.nan
    m = x.mean()
    se = stats.sem(x)
    lo, hi = stats.t.interval(1-alpha, len(x)-1, loc=m, scale=se)
    return m, lo, hi

def main(path, out):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(path, parse_dates=["appointment_date"])
    df["attended"] = 1 - df["no_show"]
    # A/B analysis: compare assigned groups, using all records as randomized units.
    g = df.groupby("group").agg(
        appointments=("appointment_id","count"),
        revenue_per_appointment=("revenue","mean"),
        no_show_rate=("no_show","mean"),
        repeat_rate=("repeat_within_60d","mean")
    ).reset_index()
    g.to_csv(os.path.join(out,"summary_metrics.csv"), index=False)

    c = df.loc[df.group=="control"]
    t = df.loc[df.group=="treatment"]
    # Welch t-test and CI for revenue per appointment
    diff = t.revenue.mean() - c.revenue.mean()
    se = np.sqrt(t.revenue.var(ddof=1)/len(t) + c.revenue.var(ddof=1)/len(c))
    dof = (t.revenue.var(ddof=1)/len(t)+c.revenue.var(ddof=1)/len(c))**2 / (
        (t.revenue.var(ddof=1)/len(t))**2/(len(t)-1) +
        (c.revenue.var(ddof=1)/len(c))**2/(len(c)-1))
    crit = stats.t.ppf(.975, dof)
    pval = stats.ttest_ind(t.revenue, c.revenue, equal_var=False).pvalue
    # SRM chi-square check against 50:50 assignment
    counts = df.group.value_counts().reindex(["control","treatment"]).fillna(0).values
    srm_chi = ((counts - counts.sum()/2)**2/(counts.sum()/2)).sum()
    srm_p = stats.chi2.sf(srm_chi, 1)
    # Proportion test for no-show rate
    no_counts = np.array([t.no_show.sum(), c.no_show.sum()])
    nobs = np.array([len(t), len(c)])
    z, no_p = proportions_ztest(no_counts, nobs)
    # CUPED-style adjustment: residualize outcome on pre-period covariate
    theta = np.cov(df.prior_revenue, df.revenue, ddof=1)[0,1] / np.var(df.prior_revenue, ddof=1)
    df["revenue_cuped"] = df.revenue - theta*(df.prior_revenue - df.prior_revenue.mean())
    tc = df.loc[df.group=="treatment","revenue_cuped"]
    cc = df.loc[df.group=="control","revenue_cuped"]
    cuped_diff = tc.mean() - cc.mean()
    cuped_p = stats.ttest_ind(tc, cc, equal_var=False).pvalue
    # Power/sample size for detecting a 5% standardized effect, 80% power, alpha 5%
    n_per_arm = TTestIndPower().solve_power(effect_size=.20, power=.80, alpha=.05, ratio=1)
    ab = pd.DataFrame([
      ["Revenue per appointment (INR)", diff, diff-crit*se, diff+crit*se, pval],
      ["CUPED-adjusted revenue (INR)", cuped_diff, np.nan, np.nan, cuped_p],
      ["No-show rate (treatment minus control)", t.no_show.mean()-c.no_show.mean(), np.nan, np.nan, no_p]
    ], columns=["metric","estimated_effect","ci95_lower","ci95_upper","p_value"])
    ab.to_csv(os.path.join(out,"ab_test_results.csv"), index=False)
    pd.DataFrame([{"control_n":counts[0],"treatment_n":counts[1],"srm_chi_square":srm_chi,
                   "srm_p_value":srm_p,"approx_n_per_arm_for_d_0_20":n_per_arm}]).to_csv(
                   os.path.join(out,"experiment_checks.csv"), index=False)

    # Difference-in-Differences: outlet-level outcome, illustrative historical rollout.
    outlet = df.groupby(["outlet_id","did_group","period"]).revenue.mean().reset_index()
    did_model = smf.ols("revenue ~ C(did_group) * C(period)", data=outlet).fit(cov_type="HC3")
    term = [x for x in did_model.params.index if ":" in x][0]
    did_effect = did_model.params[term]
    did_ci = did_model.conf_int().loc[term].tolist()
    pd.DataFrame([{"did_effect_inr":did_effect,"ci95_lower":did_ci[0],"ci95_upper":did_ci[1],
                   "p_value":did_model.pvalues[term],
                   "note":"Interpret causally only if parallel trends and no differential concurrent shocks are plausible."}]
                 ).to_csv(os.path.join(out,"did_results.csv"), index=False)

    # Segment results
    seg = df.groupby("service_category").agg(
        appointments=("appointment_id","count"),
        treatment_revenue=("revenue", lambda x: x[df.loc[x.index,"group"].eq("treatment")].mean()),
        control_revenue=("revenue", lambda x: x[df.loc[x.index,"group"].eq("control")].mean()),
        no_show_rate=("no_show","mean")
    ).reset_index()
    seg["revenue_difference"] = seg.treatment_revenue - seg.control_revenue
    seg.to_csv(os.path.join(out,"segment_results.csv"), index=False)

    # Charts
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7,4))
    sns.barplot(data=df, x="group", y="revenue", errorbar=("ci",95))
    plt.title("Average Revenue per Appointment by Group"); plt.ylabel("Revenue (INR)")
    plt.tight_layout(); plt.savefig(os.path.join(out,"fig_revenue_by_group.png"), dpi=150); plt.close()
    plt.figure(figsize=(7,4))
    sns.barplot(data=df, x="group", y="no_show")
    plt.title("No-show Rate by Group"); plt.ylabel("No-show rate")
    plt.tight_layout(); plt.savefig(os.path.join(out,"fig_no_show_by_group.png"), dpi=150); plt.close()
    plt.figure(figsize=(7,4))
    sns.pointplot(data=outlet, x="period", y="revenue", hue="group", errorbar=("ci",95))
    plt.title("Pre/Post Revenue Trends by Group"); plt.ylabel("Mean revenue (INR)")
    plt.tight_layout(); plt.savefig(os.path.join(out,"fig_did_trends.png"), dpi=150); plt.close()

    # Feature importance model is predictive/descriptive, not causal.
    X = df[["outlet_id","service_category","channel","discount_pct","prior_revenue"]]
    y = df.revenue
    cat = ["outlet_id","service_category","channel"]
    num = ["discount_pct","prior_revenue"]
    pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),
                             ("num",SimpleImputer(strategy="median"),num)])
    pipe = Pipeline([("prep",pre),("model",RandomForestRegressor(n_estimators=80,max_depth=10,
                                                                 random_state=42,n_jobs=-1))])
    sample = df.sample(min(50000,len(df)), random_state=42)
    pipe.fit(sample[ X.columns ], sample.revenue)
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    fi = pd.Series(imp,index=names).sort_values(ascending=False).head(12)
    plt.figure(figsize=(8,5)); fi.sort_values().plot(kind="barh")
    plt.title("Top Predictive Feature Importances"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(os.path.join(out,"fig_feature_importance.png"),dpi=150); plt.close()

    print("Analysis complete. Outputs saved to:", out)
    print("A/B revenue effect:", round(diff,2), "INR; p-value:", round(pval,4))
    print("SRM p-value:", round(srm_p,4), "| DiD effect:", round(did_effect,2), "INR")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/salon_appointments.csv")
    p.add_argument("--output",default="outputs")
    a=p.parse_args(); main(a.input,a.output)
