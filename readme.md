
# Overview

This repository builds on [PaperQA2](https://github.com/Future-House/paper-qa) to create a chat interface that answers questions using a collection of research papers. By integrating with Zotero, it allows efficient indexing of existing paper collections without relying on external APIs like Semantic Scholar or Crossref. The index is updated only when new papers are added minimizing redundant. This avoids dependency on API keys and improves usability for Zotero users, which is especially useful given that Semantic Scholar’s ‘API key issuance is currently on pause due to high demand.’

# Getting Started

## Installation

1. Create clean conda environment

```
mamba create -y -n paperqa-env python==3.11 pandas bibtexparser ipykernel
mamba activate paperqa-env
```

2. Install [PaperQA2](https://github.com/Future-House/paper-qa) into your conda environment

```bash
python -m pip install git+https://github.com/Future-House/paper-qa
```

3. Fork or clone this repository, set working directory (in Python session) to this directory:

```bash
git clone https://github.com/psl-schaefer/expert_system
# cd expert_system
# code .
```

## Exporting Collection from Zotero

1. In Zotero, right-click your collection and select `Export Collection`.

2. Choose a directory name (e.g., `TEST_EXPORT`), which will contain:
   - `TEST_EXPORT.bib`: Metadata file linking papers to PDFs.
   - `files/`: Folder containing all PDF files.

3. The bib file links links paper metadata to the PDFs, i.e. the `file` attribute contains the path to the corresponding PDF

```txt
@article{10.1038/s41593-024-01765-6,
title = {Spatially Resolved Gene Signatures of White Matter Lesion Progression in Multiple Sclerosis},
author = {Alsema, Astrid M. and [...] Kooistra, Susanne M. and Eggen, Bart J. L.},
year = {2024},
journal = {Nature Neuroscience},
issn = {1097-6256},
doi = {10.1038/s41593-024-01765-6},
abstract = {[...]},
pmid = {39501035},
file = {files/27939/Alsema et al. - 2024 - Spatially resolved gene signatures of white matter lesion progression in multiple sclerosis.pdf}
}
```

## Using PaperQA

- [PaperQA2](https://github.com/Future-House/paper-qa), also provides a detailed documentation here: https://github.com/Future-House/paper-qa?tab=readme-ov-file#library-usage

- Here, we follow the step from the `tutorial.ipynb` notebook.

0. Import all required packages into the python session

```python
import os
from pathlib import Path
from paperqa.settings import Settings, AgentSettings, IndexSettings, AnswerSettings
from src.build_search_index import process_bibtex_and_pdfs, create_manifest_file, build_search_index
from src.query_answer_index import query_answer_index
from src.utils import pretty_print_text
```

1. Since PaperQA uses LLMs, we either need to host one locally, or we need to provide API keys for cloud-based LLMs like DeepSeek, OpenAI, or Google Gemini. Here is a small code chunk to check whether/which API keys are set as environment variables.

```python
API_KEYS = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
for api_key in API_KEYS:
    if (key := os.getenv(api_key)):
        print(f"{api_key} found")
```

2. Use bibtex file to create a `manifest.csv` file which is required input for PaperQA

```python
# Set project-specific paths
export_directory_name = "TEST_EXPORT"
project_dir = Path(".")

data_dir = project_dir / "data"
data_dir.mkdir(exist_ok=True)
paper_directory = data_dir / export_directory_name
index_directory = data_dir / f"{export_directory_name}_index"
bibtex_file = paper_directory / f"{export_directory_name}.bib"
manifest_file = data_dir / f"{export_directory_name}_manifest.csv"
index_name = f"pqa_index_{export_directory_name}"

processed_df = process_bibtex_and_pdfs(bibtex_file=bibtex_file, paper_directory=paper_directory)
create_manifest_file(manifest_df=processed_df, manifest_file=manifest_file, paper_directory=paper_directory)
```

3. Next we need to specify all the PaperQA settings, e.g. I currently use the settings as shown below.

    - I use the `gemini/gemini-2.0-flash` LLM, because it has good performance, and is cheaper than OpenAI models.
    - Apart from that, the settings are mostly default settings.

```python
default_llm = "gemini/gemini-2.0-flash" 

index_settings = IndexSettings(
    name = index_name,
    paper_directory = paper_directory,
    manifest_file = manifest_file,
    index_directory = index_directory,
    use_absolute_paper_directory = False,
    recurse_subdirectories = True,
    concurrency = 1, # "number of concurrent filesystem reads for indexing
)

agent_settings = AgentSettings(
    agent_llm = default_llm, # smaller than default (bc cheaper)
    index = index_settings,
    index_concurrency = index_settings.concurrency
)

answer_settings = AnswerSettings(
    evidence_k = 10, # number of evidence text chunks to retrieve
    evidence_summary_length = "about 100 words", 
    answer_max_sources = 5, # max number of sources to use for answering
    answer_length = "about 200 words, but can be longer", # length of final answer
)

settings = Settings(
    agent = agent_settings, 
    answer = answer_settings,
    llm=default_llm,
    summary_llm=default_llm, 
    embedding="text-embedding-3-small", # default
    temperature = 0.0, # default
    texts_index_mmr_lambda = 1.0, # Lambda MMR
    index_absolute_directory = index_settings.use_absolute_paper_directory,
    index_directory = index_settings.index_directory,
    index_recursively = index_settings.recurse_subdirectories,
    manifest_file = index_settings.manifest_file,
    paper_directory = index_settings.paper_directory,
    verbosity = 0, # (0-3), -> 3 is all LLM/Embedding calls are logged
)
```

4. Then after specifying the PaperQA settings, we are ready to chunk the PDFs and embed the text chunks in a vector database. 

    - This step can take 10-30 minutes (depending on the number of documents), since a text-embedding model (here: OpenAI model `text-embedding-3-small`) must be called for each text chunk

```python
search_index = await build_search_index(settings=settings, bibtex_file=bibtex_file, manifest_file=manifest_file)
assert search_index.index_name == settings.agent.index.name
print(f"Index Name: {search_index.index_name}")
print(f"Number of Indexed Files: {len((await search_index.index_files).keys())}")
```

```txt
Index Name: pqa_index_TEST_EXPORT
Number of Indexed Files: 10
```

5. Finally, we are ready to ask questions!

```python
answer_response = await agent_query(
    query="What is the role of the perivascular niche in multiple sclerosis?", 
    settings=settings, rebuild_index=False
)
pretty_print_text(answer_response.session.formatted_answer)
```

```txt
Question: What is the role of the perivascular niche in multiple sclerosis?

The perivascular niche plays a critical role in the pathogenesis of multiple
sclerosis (MS) by facilitating immune cell trafficking and inflammation in the
central nervous system (CNS). Immune cells, including T cells (CD4+ and CD8+),
B cells, and monocytes, migrate across the blood-brain barrier (BBB) at
post-capillary venules, entering the CNS parenchyma and contributing to
perivascular lesion formation (Filippi2021 pages 6-7, Filippi2021 pages 8-9).

...

1. (Filippi2021 pages 6-7): Filippi, Massimo, et al. "Multiple Sclerosis."
*Nature Reviews Disease Primers*, vol. 7, no. 1, 2021, article 1. Accessed
2024

...
```

6. All answers and responses are by saved in the answer index which we can query

```python
query_answer_index_results = await query_answer_index(settings=settings, query="role of perivascular niche in MS")
```

# Notes

- If you do not use Zotero you need to create a bibtex file that - just like the bibtex file shown above - contains a file attribute which provides the relative path to the corresponding PDF files.

# TODO

- [ ] Improve handling of multipe PDFs per Zotero (metadata) entry:

    - Currently, if all PDFs have the same length, I simply use the first one

    - If the lengths are different, I merge the PDFs
