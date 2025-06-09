# Toxicity detection tool for Eredivisie Twitter communities (NLP + SNA)

This repository contains a software tool that combines Natural Language Processing (NLP) and Social Network Analysis (SNA) to detect, track, and visualize the spread of toxicity in Dutch Eredivisie-related Twitter communities during emotionally charged football events.

The project was developed as part of a Master’s thesis at the University of Applied Sciences Amsterdam and is specifically focused on tool development — not on analyzing or interpreting the social behavior itself. The system provides a modular pipeline for data collection, toxicity classification, network construction, and interactive visualization.

---

## Thesis research question

**"How can a software tool that combines NLP and Social Network Analysis be developed to detect, track, and visualize the spread of toxicity in Eredivisie-related Twitter communities during emotionally charged football events?"**

---

## Repository structure

**Data/**
- `Timelines/` – Original tweets from club accounts
- `Replies/` – First-level replies scraped from Nitter
- `Second_level_replies/` – Reply-to-reply threads
- `final_sna_dataframe.csv` – Cleaned and combined dataset for analysis
- `toxic_match_metadata_cleaned.csv` – Metadata on matches and trigger events
- Other `.csv` and `.pkl` files – Intermediate data for NLP and SNA steps

**Notebooks/**
- `NLP_Notebook.ipynb` – Toxicity classification pipeline
- `SNA_TestV2.ipynb` – Full SNA pipeline with interactive dashboard
- `merge_scraped_tweets.ipynb` – Merges multi-layered tweet data
- `prepare_match_metadata.ipynb` – Cleans and aligns match metadata

**Scraper/**
- `Eredivisie_scraperV2.py` – Selenium scraper for Nitter (non-API access)

**Document/**
- `Thesis report` – Final report and/or visual materials

---

## Overview of the tool

This tool supports the full pipeline for detecting and visualizing toxicity in social media networks tied to Dutch football matches. It includes:

- **Data collection** from Nitter using a custom Selenium-based scraper
- **Toxicity classification** of tweets using a pre-trained Dutch BERT model
- **Reply network generation** per match and per club
- **Community detection** using modularity algorithms
- **Node labeling** based on toxicity ratio thresholds
- **Interactive visualization dashboard** built with `ipywidgets`, `networkx`, and `plotly`

---

## Dataset origin

The dataset used in this tool is a manually compiled list of all Eredivisie football matches played during the 2024–2025 season. It includes key metadata for each match such as:

- Match ID
- Date and time (UTC)
- Home and away clubs
- Toxic event type

This dataset is stored in `Matchweeks_Eredivisie V2.xlsx` and is used as input for the notebook `prepare_match_metadata.ipynb`. That notebook cleans and transforms the raw match schedule into `toxic_match_metadata_cleaned.csv`, which is required for all subsequent steps in the pipeline.

> ⚠️ This is a required starting point. Without this dataset, the scraping and analysis pipeline will not work.

If you'd like to reproduce or adapt the dataset for a different season, you can manually construct a similar Excel file using public match schedules from sources like:
- [eredivisie.nl](https://eredivisie.nl)
- [sofascore.com](https://www.sofascore.com/)
- [espn.com/soccer](https://www.espn.com/soccer/)

---

## How to use

1. Clone the repository:
   ```bash
   git clone https://github.com/MasterDDB24/mpTim.git

2. Make a virtual environment with the packages and versions listed in `requirements.txt`, and install them:
   ```bash
   pip install -r requirements.txt

3. Run the notebooks in this order:
   - `prepare_match_metadata.ipynb`  
     > **Note:** This notebook generates the `toxic_match_metadata_cleaned.csv` file required by the scraper. Make sure it completes successfully before moving on.
   
   - `Eredivisie_scraperV2.py`  
     > This script depends on the metadata file created in the previous step. It assumes that `toxic_match_metadata_cleaned.csv` is saved in `C:/Master/Master project/`. No command-line arguments are needed — you can run the script directly.
   
   - `merge_scraped_tweets.ipynb`  
     > This notebook merges all scraped tweet timelines and replies into a single cleaned dataset for further analysis. Ensure the `timelines/`, `replies/`, and `second_level_replies/` folders exist and are populated before running this step.
   
   - `NLP_Notebook.ipynb`  
     > This notebook applies toxicity classification to all merged tweets using a pre-trained model. It adds toxicity scores and categories to the dataset, preparing it for network analysis.
   
   - `SNA_TestV2.ipynb`  
     > This is the final analysis notebook. It builds the social network graphs, applies community detection, calculates network metrics, and provides an interactive dashboard to explore toxicity patterns across matches and clubs.

The final notebook contains the interactive dashboard where you can visualize toxicity across matchdays and clubs, filter by threshold, explore communities, and export figures.

---

## Output capabilities

- Match-level and club-level reply networks
- Toxicity-based node coloring and labeling
- Community-level toxicity breakdowns
- Timeline plots with match event overlays
- Interactive visual interface with threshold slider, layer switching, and statistics

---

## Environment

All experiments were conducted on a local machine with the following hardware specifications:

- CPU: 13th Gen Intel Core i7-1355U
- RAM: 32 GB
- GPU: Integrated Intel Iris Xe Graphics (no CUDA support)
- Storage: 1 TB SSD (NVMe)
- OS: Windows 11
- Python: 3.10
- PyTorch: 2.0 (CPU-only)

Note: All model inference and training steps in this notebook were executed **on CPU only**, as the system does not have a CUDA-compatible GPU. While the pipeline is designed to support GPU acceleration, it remains fully functional on CPU — although inference and training runtimes may be significantly longer.

---

## Notes

- This is a tool development project. The code and logic are designed to support flexible re-use on other Twitter/Nitter datasets.
- Raw scraped tweet data is included under `Data/` in separate folders by type.
- The tool is modular and extendable for future use with additional toxicity models or languages.

---

## Author

Tim Zijlstra  
Master’s Thesis 2025  
University of Applied Sciences Amsterdam – Digital Driven Business  
Thesis type: Tool development
