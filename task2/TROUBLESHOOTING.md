# Troubleshooting (Windows + VS Code)

| Problem you see | Why it happens | Fix |
|---|---|---|
| `Could not open requirements file: 'requirements.txt'` | The zip unzipped into a folder inside a folder, and the terminal is in the outer one | Run `dir`. If you see the project folder inside, `cd` into it until `requirements.txt` is listed |
| `OSError ... No such file or directory ... \.venv\Lib\site-packages\...` + "Long Path support" hint | The folder path is too long for Windows (260 character limit) | Move the project to a short path such as `C:\dt` (`robocopy "<old path>" C:\dt /E /XD .venv`), then create `.venv` again there |
| `python is not recognized` | Python is not on PATH | Reinstall Python and tick "Add Python to PATH", or use `py` instead of `python` |
| `.venv\Scripts\activate` is blocked (red PowerShell error) | PowerShell script policy | `Set-ExecutionPolicy -Scope Process Bypass`, then activate again |
| `ModuleNotFoundError` inside the notebook | Notebook is using the wrong Python | Top-right **Select Kernel** -> **Python Environments** -> choose the `.venv` inside this project |
| App shows "model file is missing" | The notebook has not finished, so `models/` was never created | Run the whole notebook (Run All) and check `dir models` |
| `UnicodeEncodeError: 'charmap' codec ...` | Windows writes files in a legacy encoding | Already fixed in this version of `finalize.py`. On an old copy run `$env:PYTHONUTF8 = "1"` first |
| Demo works locally but fails on Streamlit Cloud with a scikit-learn / pickle error | Cloud installed a different library version than the one that trained the model | Run `python freeze_requirements.py` (finalize.py does it for you), commit `requirements.txt` and push again |
| Cloud app says model file not found | `models/` was not pushed | Run `git add -f models` then commit and push |
