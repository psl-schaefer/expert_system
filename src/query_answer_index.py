from paperqa.settings import Settings
from paperqa.agents.search import SearchIndex, SearchDocumentStorage
from paperqa.agents.models import AnswerResponse


async def query_answer_index(
        settings: Settings,
        query: str
    ):
    """
    Querying the answers using tantivy (so query is not embedded by embedding model)
    """
    answers_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "question"],
        index_name="answers",
        index_directory=settings.agent.index.index_directory,
        storage=SearchDocumentStorage.JSON_MODEL_DUMP,
    )
    print(f"Number of Indexed Answers: {len((await answers_index.index_files).keys())}")
    results = [AnswerResponse(**a[0]) for a in (await answers_index.query(query=query, keep_filenames=True))]
    print(f"Number of Answers Matching Query: {len(results)}")
    return results