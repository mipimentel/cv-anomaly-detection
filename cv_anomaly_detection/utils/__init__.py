import os

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "datasets")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MVTEC_AD = os.path.join(DATA_DIR, "mvtec")
# MODELS_DIR = os.path.join(BASE_DIR, 'models')

# translation dict for portuguese plots
# df["class"] = df["class"].map(translation_dict)
translation_dict = {
    "good": "conforme",
    "bent_wire": "fio dobrado",
    "missing_wire": "fios faltantes",
    "missing_cable": "cabo faltante",
    "cut_inner_insulation": "isolamento interno cortado",
    "cut_outer_insulation": "isolamento externo cortado",
    "poke_insulation": "isolamento furado",
    "cable_swap": "cabo trocado",
    "combined": "defeitos mistos",
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MVTEC_AD, exist_ok=True)
