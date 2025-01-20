
import os
import anyio
import re
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import bibtexparser
import pymupdf

from paperqa.settings import Settings
from paperqa.agents.search import SearchIndex
from paperqa.utils import maybe_is_text, md5sum
from paperqa.docs import Docs, Doc, DocDetails, read_doc

# see also: format_bibtex, populate_bibtex_key_citation
BIBTEX_ATTR = ["doi", "authors", "title", "year", "publisher", "issn", "volume", "pages", "journal"]
WARN_IF_INDEXING_MORE_THAN = 999


logger = logging.getLogger(__name__)


async def build_search_index(
        settings: Settings,
        bibtex_file: os.PathLike,
        manifest_file: os.PathLike,
        force_reindexing: bool = False
) -> SearchIndex:
    manifest_df = pd.read_csv(manifest_file)

    with open(bibtex_file, "r") as bibfile:
        bib_database = bibtexparser.load(bibfile)
    bib_list = bib_database.entries
    # NOTE: We only use PDF for now!
    bib_dict = {re.search(r'[^;]*\.pdf', entry["file"]).group(0): entry for entry in bib_list}

    index_settings = settings.agent.index
    # NOTE: We assume that we have already named the index!
    assert index_settings.name is not None

    # TODO: I did not test it with abbsolute, so let's forbid for now
    assert not index_settings.use_absolute_paper_directory

    search_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        index_name=index_settings.name,
        index_directory=index_settings.index_directory,
    )

    # get the files that have already been indexed
    already_indexed_files = await search_index.index_files

    paper_directory = anyio.Path(index_settings.paper_directory)

    valid_papers_rel_file_paths = [
        file.relative_to(paper_directory)
        async for file in (
            paper_directory.rglob("*")
            if index_settings.recurse_subdirectories
            else paper_directory.iterdir()
        )
        # NOTE 1: I only include PDFs here and not {".txt", ".pdf", ".html"}
        # NOTE 2: I only include PDFs that are in my manifest file
        if (file.suffix in {".pdf"}) and (str(file) in manifest_df["file_location"].to_numpy())
    ]
    logger.info(f"Found {valid_papers_rel_file_paths} Valid PDFs")

    if len(valid_papers_rel_file_paths) > WARN_IF_INDEXING_MORE_THAN:
        logger.info(
            f"Indexing {len(valid_papers_rel_file_paths)} files into the index"
            f" {search_index.index_name}, may take a few minutes."
        )

    index_unique_file_paths: set[str] = set((await search_index.index_files).keys())
    if extra_index_files := (
        index_unique_file_paths - {str(f) for f in valid_papers_rel_file_paths}
    ):
        if index_settings.sync_with_paper_directory:
            for extra_file in extra_index_files:
                logger.warning(
                    f"[bold red]Removing {extra_file} from index.[/bold red]"
                )
                await search_index.remove_from_index(extra_file)
            logger.warning("[bold red]Files removed![/bold red]")
        else:
            logger.warning(
                f"[bold red]Indexed files {extra_index_files} are missing from paper"
                f" folder ({paper_directory}).[/bold red]"
            )

    # Skip all files that have already been indexed
    n_before = len(valid_papers_rel_file_paths)
    if not force_reindexing:
        valid_papers_rel_file_paths = [p for p in valid_papers_rel_file_paths if str(p) not in already_indexed_files]
        n_after = len(valid_papers_rel_file_paths)
        n_skipped = n_before - n_after
        logger.info(f"Indexing {n_after} documents, {n_skipped} skipped as they were already indexed.")
    else:
        n_could_be_skipped = sum(1 for p in valid_papers_rel_file_paths if str(p) in already_indexed_files)
        logger.info(f"Indexing all {n_before} documents without skipping {n_could_be_skipped} previously indexed documents.")

    for rel_file_path in valid_papers_rel_file_paths:

        # NOTE: This corresponds to the original code from `get_directory_index``
        #await process_file(rel_file_path=rel_file_path, manifest=manifest, search_index=search_index)
        # thus the code below is adapted from `get_directory_index`

        logger.debug(f"Start indexing {rel_file_path}")

        docs_for_single_doc = Docs() # otherwise creating Docs object for single doc confuses me

        abs_file_path = settings.paper_directory / rel_file_path

        bib_dict_doc = bib_dict[str(rel_file_path)]
        if "author" in bib_dict_doc.keys():
            bib_dict_doc["authors"] = _format_authors(bib_dict_doc["author"])
            bib_dict_doc["authors"] = [auth.replace("{", "").replace("}", "") for auth in bib_dict_doc["authors"]]

        parse_config = settings.parsing
        dockey = md5sum(abs_file_path)

        llm_model = settings.get_llm()

        texts = read_doc(
                abs_file_path,
                Doc(docname="", citation="", dockey=dockey),  # Fake doc
                chunk_chars=parse_config.chunk_size,
                overlap=parse_config.overlap,
                page_size_limit=parse_config.page_size_limit,
            )

        if not texts:
            raise ValueError(f"Could not read document {abs_file_path}. Is it empty?")

        result = await llm_model.run_prompt(
            prompt=parse_config.citation_prompt,
            data={"text": texts[0].text},
            system_prompt=None,  # skip system because it's too hesitant to answer
        )
        citation = result.text

        if (
            len(citation) < 3  # noqa: PLR2004
            or "Unknown" in citation
            or "insufficient" in citation
        ):
            citation = f"Unknown, {os.path.basename(abs_file_path)}, {datetime.now().year}"

        # get first name and year from citation
        match = re.search(r"([A-Z][a-z]+)", citation)
        if match is not None:
            author = match.group(1)
        else:
            # panicking - no word??
            raise ValueError(
                f"Could not parse docname from citation {citation}. "
                "Consider just passing key explicitly - e.g. docs.py "
                "(path, citation, key='mykey')"
            )
        year = ""
        match = re.search(r"(\d{4})", citation)
        if match is not None:
            year = match.group(1)
        docname = f"{author}{year}"
        docname = docs_for_single_doc._get_unique_name(docname)

        doc = Doc(docname=docname, citation=citation, dockey=dockey)

        # see also CROSSREF_API_MAPPING, SEMANTIC_SCHOLAR_API_MAPPING
        doc_details_dict = {k: bib_dict_doc[k] for k in BIBTEX_ATTR if k in bib_dict_doc.keys()}
        doc_details = DocDetails(**doc_details_dict)
        doc_details.dockey = doc.dockey
        doc_details.doc_id = doc.dockey
        doc_details.docname = doc.docname
        doc_details.key = doc.docname
        doc_details.citation = doc.citation

        # I am not checking this, but one could use CrossRef for this!
        doc_details.is_retracted = False
        doc_details.file_location = abs_file_path
        doc_details.doc_id = bib_dict_doc["doi"] if "doi" in bib_dict_doc.keys() else None

        embedding_model = settings.get_embedding_model()

        texts = read_doc(
            abs_file_path,
            doc,
            chunk_chars=parse_config.chunk_size,
            overlap=parse_config.overlap,
            page_size_limit=parse_config.page_size_limit,
        )
        # loose check to see if document was loaded
        if (
            not texts
            or len(texts[0].text) < 10  # noqa: PLR2004
            or (
                not parse_config.disable_doc_valid_check
                # Use the first few text chunks to avoid potential issues with title page parsing in the first chunk
                and not maybe_is_text("".join(text.text for text in texts[:5]))
            )
        ):
            raise ValueError(
                f"This does not look like a text document: {abs_file_path}. Pass disable_check"
                " to ignore this error."
            )
        _ = await docs_for_single_doc.aadd_texts(texts=texts, doc=doc, settings=settings, embedding_model=embedding_model)

        # also add to the search index
        fallback_title = rel_file_path.name
        file_location = str(rel_file_path) # important to use the relative path here, as the script compares hashes
        if isinstance(doc, DocDetails):
            title = doc.title or fallback_title
            year = doc.year or "Unknown year"
        else:
            title, year = fallback_title, "Unknown year"

        await search_index.add_document(
            {
                "title": title,
                "year": year,
                "file_location": file_location,
                "body": "".join(t.text for t in docs_for_single_doc.texts),
            },
            document=docs_for_single_doc,
        )

        # Save so we can resume the build without rebuilding this file
        # TODO: Check if saving it after adding one file adds to mcuh overhead
        await search_index.save_index()
        
    return search_index



def _format_authors(bibtex_authors):
    # Split authors by "and"
    authors = [author.strip() for author in bibtex_authors.split(" and ")]
    
    # Reformat each author from "Last, First" to "First Last"
    formatted_authors = []
    for author in authors:
        if ',' in author:
            last, first = map(str.strip, author.split(',', 1))
            formatted_authors.append(f"{first} {last}")
        else:
            formatted_authors.append(author)  # In case no comma, keep as is
    
    return formatted_authors


def process_bibtex_and_pdfs(bibtex_file: os.PathLike, paper_directory: os.PathLike) -> pd.DataFrame:
    """
    Parses a BibTeX file and processes associated PDFs.

    Args:
        bibtex_file (os.PathLike): Path to the BibTeX file.
        paper_directory (os.PathLike): Directory containing the papers.

    Returns:
        pd.DataFrame: DataFrame with columns `title`, `doi`, and `file_location`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    attributes = ["title", "doi", "file"]
    processed_entries = []

    with open(bibtex_file, "r") as bibfile:
        bib_database = bibtexparser.load(bibfile)

    bib_list = bib_database.entries
    logging.info(f"Total number of entries in the BibTeX file: {len(bib_list)}")

    for entry in bib_list:
        check_vec = np.array([attribute in entry.keys() for attribute in attributes])
        if not np.all(check_vec):
            missing_attrs = [attr for attr, present in zip(attributes, check_vec) if not present]
            logging.warning(f"Entry missing attributes {missing_attrs}: {entry.get('title', 'Unknown title')}")
            continue

        file_location = entry["file"]
        pdf_parts = [part for part in file_location.split(";") if part.endswith(".pdf")]

        if len(pdf_parts) == 0:
            logging.warning(f"No PDF found for: {entry['title']}")
            continue

        if len(pdf_parts) == 1:
            # Only one PDF, use it directly
            selected_pdf = paper_directory / Path(pdf_parts[0])
        else:
            # Multiple PDFs, process them
            pdf_paths = [str(paper_directory / Path(pdf)) for pdf in pdf_parts]
            valid_pdfs = [pdf for pdf in pdf_paths if os.path.exists(pdf)]

            if len(valid_pdfs) < len(pdf_parts):
                missing_files = [p for p in pdf_paths if not os.path.exists(p)]
                logging.warning(f"Some PDFs listed do not exist: {missing_files}")

            if len(valid_pdfs) == 0:
                logging.warning(f"No valid PDFs for entry: {entry['title']}")
                continue

            # Check PDF lengths
            pdf_lengths = [_get_page_count(pdf) for pdf in valid_pdfs]
            if all(length == pdf_lengths[0] for length in pdf_lengths):
                # If all lengths are the same, keep the first PDF
                selected_pdf = valid_pdfs[0]
                logging.info(f"Kept first PDF for: {entry['title']}")
            else:
                # Merge PDFs if lengths differ
                merged_pdf_path = str(paper_directory / f"{Path(valid_pdfs[0]).stem}_merged.pdf")
                _merge_pdfs(valid_pdfs, merged_pdf_path)
                selected_pdf = merged_pdf_path
                logging.info(f"Merged PDFs for: {entry['title']}")

        processed_entries.append({
            "title": entry["title"],
            "doi": entry["doi"],
            "file_location": str(Path(selected_pdf))
        })

    return pd.DataFrame(processed_entries)


def create_manifest_file(manifest_df: pd.DataFrame, manifest_file: os.PathLike):
    """
    Creates a manifest file from a DataFrame.

    Args:
        manifest_df (pd.DataFrame): DataFrame containing `title`, `doi`, and `file_location`.
        manifest_file (os.PathLike): Path to save the manifest file.
    """
    # Check existence of files and log missing ones
    file_exists = manifest_df["file_location"].apply(os.path.exists)
    for idx, exists in enumerate(file_exists):
        if not exists:
            logging.warning(f"{manifest_df.loc[idx, 'file_location']} does not exist")

    logging.info(f"Number of entries with existing files: {file_exists.sum()}")
    manifest_df.to_csv(manifest_file, index=False)
    logging.info(f"Manifest file written to {manifest_file}")


def _get_page_count(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        return len(doc)


def _merge_pdfs(pdf_paths, output_path):
    merged_doc = pymupdf.open()
    for pdf in pdf_paths:
        with pymupdf.open(pdf) as doc:
            merged_doc.insert_pdf(doc)  # Append each PDF
    merged_doc.save(output_path)