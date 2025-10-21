import time
import requests
import pandas as pd
from tqdm.auto import tqdm 

def download_list_of_pathways(save_dir: str = "./"):
    """Download the master list of Reactome pathways.

    Fetches the Reactome pathway list (ReactomePathways.txt) from the Reactome
    database and saves it locally as a text file.

    Args:
        save_dir (str, optional): Directory path to save the file. Defaults to "./".
    """
    res = requests.get('https://reactome.org/download/current/ReactomePathways.txt')
    if res.ok:
        with open(f"{save_dir}list_of_pathways.txt", mode="wb") as file:
            file.write(res.content)
    else:
        ValueError(res)

def download_single_biopax_file_by_pathway_id(id: str, save_dir: str = "./"):
    """Download a single BioPAX Level 3 file by Reactome pathway ID.

    Fetches the BioPAX XML file corresponding to the given Reactome pathway ID
    from the Reactome REST API and saves it locally.

    Args:
        id (str): Reactome pathway ID (e.g., "R-HSA-199420").
        save_dir (str, optional): Directory path to save the BioPAX XML file. Defaults to "./".
    """
    url = f"https://reactome.org/ReactomeRESTfulAPI/RESTfulWS/biopaxExporter/Level3/{id.strip('R-HSA-')}"
    res = requests.get(url, stream=True)
    if res.ok:
        with open(f'{save_dir}{id}.xml', 'wb') as out_file:
            out_file.write(res.content)
    else:
        ValueError(res)

def download_biopax_files_by_org(save_dir: str = "./", species: str = "Homo sapiens", sleep_timer: int = 30):
    """Download BioPAX Level 3 files for all pathways of a given organism.

    Reads the list of Reactome pathways, filters by species name, and downloads
    each corresponding BioPAX file sequentially. Retries failed downloads after
    a sleep delay.

    Args:
        save_dir (str, optional): Directory path to save BioPAX files. Defaults to "./".
        species (str, optional): Species name to filter pathways (e.g., "Homo sapiens"). Defaults to "Homo sapiens".
        sleep_timer (int, optional): Delay (in seconds) before retrying a failed download. Defaults to 30.
    """
    list_of_pathways = pd.read_csv("./data/list_of_pathways.txt", delimiter="\t", names = ["id", "name", "species"])
    list_of_pathways = list_of_pathways[list_of_pathways.species == species]
    for ids in tqdm(list_of_pathways.id):
        try:
            download_single_biopax_file_by_pathway_id(ids, save_dir)
        except Exception as e:
            print(e)
            time.sleep(sleep_timer)
            download_single_biopax_file_by_pathway_id(ids, save_dir)