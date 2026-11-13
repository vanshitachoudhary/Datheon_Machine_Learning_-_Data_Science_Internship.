# Step-by-step: run, push to GitHub, deploy, post

## 1. Run the notebook
```bash
pip install -r requirements-dev.txt
```
Tip: keep the folder path short (for example `C:\dt`). Very long Windows paths make `pip install` fail.

Open `task2_housing_regression.ipynb` in VS Code (top-right **Select Kernel** -> your `.venv`) and press **Run All**. Or run it from the terminal:
```bash
python -m nbconvert --to notebook --execute --inplace task2_housing_regression.ipynb
```
Wait until the last cell finishes. This creates `models/`, `images/` and `RESULTS.md`.

**Then save the notebook (Ctrl+S / Cmd+S).** This stores the charts and outputs inside the `.ipynb`, so evaluators see them on GitHub without running anything.

## 2. Check these exist
`models/` (one `.joblib` file), `images/` (png plots), `RESULTS.md`.

3. **Fill README + LinkedIn post automatically:**
   ```bash
   python finalize.py
   ```
   This writes your real results, conclusion and plots into `README.md` and `LINKEDIN_POST.md`, and pins `requirements.txt` to the exact library versions you trained with (so the deployed demo can load your model).

## 4. Push to GitHub
1. On github.com click **New repository**. Name: `datheon-task2-housing-price-prediction`. Set **Public**. Do **not** tick "Add a README".
2. In the project folder run (replace `YOUR-USERNAME`):
```bash
git init
git add .
git commit -m "Task 2: California Housing Price Prediction"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/datheon-task2-housing-price-prediction.git
git push -u origin main
```
3. Make sure `models/` shows up on GitHub (the demo needs the `.joblib` file). If GitHub asks for a password, use a Personal Access Token, or `gh auth login`, or GitHub Desktop.

## 5. Deploy the live demo (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** -> pick repo `datheon-task2-housing-price-prediction`, branch `main`, main file `app.py`.
3. In Advanced settings choose Python 3.11, then **Deploy**. The first build takes a few minutes.
4. Copy the app link (`https://....streamlit.app`).

   Then add the links and push again:
   ```bash
   python finalize.py --demo <your-streamlit-url> --github <your-github-repo-url>
   git add . && git commit -m "Add demo and results links" && git push
   ```

## 6. LinkedIn
1. Open `LINKEDIN_POST.md`, copy the post.
2. Attach 2-3 images from `images/` (plus a screenshot of the live demo).
3. Tag **@Datheon** (type @ and select the page) and post publicly.
4. Submit your GitHub link and the post link on the Datheon portal.
