import anndata as ad
import scanpy as sc
from pathlib import Path

DATA_DIR = Path(__file__).parent / "Demo_obj"
PUBLIC_DATASETS = {
    "Unselected": None,  
    "Microglia_demo": {
        "path": DATA_DIR / "Microglia.h5ad",
        "description": "Microglia dataset",
        "loader": lambda path: ad.read_h5ad(path)
    }
}