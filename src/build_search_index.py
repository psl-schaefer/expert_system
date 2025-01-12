
import os
import anyio
import re
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import bibtexparser

from paperqa.settings import Settings
from paperqa.agents.search import SearchIndex
from paperqa.utils import maybe_is_text, md5sum
from paperqa.docs import Docs, Doc, DocDetails, read_doc


WARN_IF_INDEXING_MORE_THAN = 999


logger = logging.getLogger(__name__)


async def build_search_index(
        settings: Settings,
        bibtex_file: os.PathLike,
        force_reindexing: bool = False
) -> SearchIndex:
    with open(bibtex_file, "r") as bibfile:
        bib_database = bibtexparser.load(bibfile)
    bib_list = bib_database.entries
    # NOTE: We onnly use PDF for now!
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
        if file.suffix in {".pdf"} # NOTE: I only include PDFs here and not {".txt", ".pdf", ".html"}
    ]

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

        doc_details = DocDetails(**{k: bib_dict_doc[k] for k in ["doi", "authors", "title", "year", "publisher", "issn", "pages", "journal"] if k in bib_dict_doc.keys()})
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

    # Save so we can resume the build without rebuilding this file if a
    # separate process_file invocation leads to a segfault or crash
    await search_index.save_index()
    
    return search_index


def manifest_from_bibtex(
        bibtex_file: os.PathLike,
        paper_directory: os.PathLike,
        manifest_file: os.PathLike
):
    attributes = ["title", "doi", "file"]
    manifest_df_list = []

    with open(bibtex_file, "r") as bibfile:
        bib_database = bibtexparser.load(bibfile)

    bib_list = bib_database.entries
    for entry in bib_list:
        check_vec = np.array([attribute in entry.keys() for attribute in attributes])
        if np.all(check_vec):
            entry_df = pd.DataFrame({"title": [entry["title"]],
                                    "doi": [entry["doi"]],
                                    "file_location": [entry["file"]]})
            manifest_df_list.append(entry_df)
        else:
            print(entry)
            print({attr: check for attr, check in zip(attributes, check_vec)})
    manifest_df = pd.concat(manifest_df_list, axis=0)
    manifest_df = manifest_df.reset_index()
    manifest_df["file_location"] = [str(paper_directory / Path(f)) for f in manifest_df["file_location"]]
    manifest_df.to_csv(manifest_file)


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