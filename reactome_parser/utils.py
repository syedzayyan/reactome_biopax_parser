import time
import requests
import pandas as pd
from tqdm.auto import tqdm 

def download_list_of_pathways(save_dir: str = "./"):
    res = requests.get('https://reactome.org/download/current/ReactomePathways.txt')
    if res.ok:
        with open(f"{save_dir}list_of_pathways.txt", mode="wb") as file:
            file.write(res.content)
    else:
        ValueError(res)

def download_single_biopax_file_by_pathway_id(id: str, save_dir: str = "./"):
    url = f"https://reactome.org/ReactomeRESTfulAPI/RESTfulWS/biopaxExporter/Level3/{id.strip('R-HSA-')}"
    res = requests.get(url, stream=True)
    if res.ok:
        with open(f'{save_dir}{id}.xml', 'wb') as out_file:
            out_file.write(res.content)
    else:
        ValueError(res)

def download_biopax_files_by_org(save_dir: str = "./", species: str = "Homo sapiens", sleep_timer: int = 30):
    list_of_pathways = pd.read_csv(f"./data/list_of_pathways.txt", delimiter="\t", names = ["id", "name", "species"])
    list_of_pathways = list_of_pathways[list_of_pathways.species == species]
    for ids in tqdm(list_of_pathways.id):
        try:
            download_single_biopax_file_by_pathway_id(ids, save_dir)
        except:
            time.sleep(sleep_timer)
            download_single_biopax_file_by_pathway_id(ids, save_dir)