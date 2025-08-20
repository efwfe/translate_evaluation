import pathlib


BASE_DIR = pathlib.Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)    

