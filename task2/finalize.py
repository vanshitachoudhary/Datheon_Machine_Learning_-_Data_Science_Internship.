"""Run AFTER the notebook:  python finalize.py [--demo STREAMLIT_URL] [--github REPO_URL]
Fills README.md and LINKEDIN_POST.md from results.json (written by the notebook)."""
import json, re, glob, os, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--demo", default=None); ap.add_argument("--github", default=None)
a = ap.parse_args()
assert os.path.exists("results.json"), "results.json not found. Run the notebook top to bottom first (Run All)."
R = json.load(open("results.json", encoding="utf-8"))
cv = {r["Model"]: r["CV RMSE ($) mean"] for r in R["cv"]}
best = R["best"]; bt = next(r for r in R["test"] if r["Model"] == best)

rows = []
for r in R["test"]:
    m = r["Model"]; name = f"**{m}**" if m == best else m
    rows.append(f"| {name} | {r['RMSE ($)']:,.0f} | {r['MAE ($)']:,.0f} | {r['R2']:.3f} | {cv[m]:,.0f} |")
table = "| Model | Test RMSE ($) | Test MAE ($) | Test R2 | 5-fold CV RMSE ($) |\n|---|---|---|---|---|\n" + "\n".join(rows)
imgs = sorted(glob.glob("images/fig*.png"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
block = ("<!-- RESULTS_START -->\n## Results\n\n"
         f"California housing districts: {R['n_rows']:,}. Best model by cross-validation: **{best}**.\n\n{table}\n\n"
         f"### Conclusion\n{R['conclusion']}\n\n### Plots\n" + "\n\n".join(f"![{os.path.basename(p)}]({p})" for p in imgs) + "\n<!-- RESULTS_END -->")

readme = open("README.md", encoding="utf-8").read()
if "<!-- RESULTS_START -->" in readme:
    readme = re.sub(r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->", lambda m: block, readme, flags=re.S)
elif "**Result (fill after running):**" in readme:
    readme = re.sub(r"\*\*Result \(fill after running\):\*\*.*?fig1\.png\)`\.\n", lambda m: block + "\n", readme, flags=re.S)
else:
    readme += "\n" + block + "\n"
if a.demo:
    readme = re.sub(r"\*\*Live demo:\*\*.*", lambda m: f"**Live demo:** {a.demo}", readme)
open("README.md", "w", encoding="utf-8").write(readme)

top = ", ".join(R["top_features"])
post = f"""Just finished Task 2 of my Machine Learning & Data Science Internship with @Datheon! 🏠

Project: California Housing Price Prediction ({R['n_rows']:,} districts)
✅ Feature engineering (ratios + interaction features), missing-value imputation, categorical encoding
✅ Checked multicollinearity with VIF ({len(R['vif_high'])} features had VIF > 10) and compared OLS vs Ridge vs Lasso with GridSearchCV
✅ Best model (5-fold CV): {best}
✅ Test RMSE ${bt['RMSE ($)']:,.0f} | MAE ${bt['MAE ($)']:,.0f} | R2 {bt['R2']:.2f}
✅ Biggest price drivers: {top}
✅ Deployed a live price predictor with Streamlit

🔗 Live demo: {a.demo or '<add link>'}
💻 GitHub: {a.github or '<add link>'}

#MachineLearning #DataScience #Python #ScikitLearn #Regression #Datheon
"""
open("LINKEDIN_POST.md", "w", encoding="utf-8").write("# LinkedIn post (ready to copy). Attach 2-3 images from images/\n\n" + post)
print("README.md and LINKEDIN_POST.md updated.")
if not a.demo or not a.github:
    print("Tip: after deploying, re-run with  --demo <streamlit link> --github <repo link>  to fill the links.")

try:
    import freeze_requirements
    freeze_requirements.main()
except Exception as e:
    print("Could not pin requirements.txt automatically:", e, "- run: python freeze_requirements.py")
