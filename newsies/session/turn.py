"""
newsies.session.turn

"""

from typing import Dict, List

from newsies import actions
from newsies import targets
from newsies.chromadb_client import ChromaDBClient

# pylint: disable=broad-exception-caught, protected-access, unnecessary-lambda, too-many-instance-attributes


def paginate_dict(large_dict: Dict, page_size: int = 100) -> List[Dict]:
    """
    paginate_dict

    Splits a large dictionary into a list of smaller dictionaries,
        each with a maximum of `page_size` keys.

    :param large_dict: The original large dictionary to be paginated.
    :param page_size: The maximum number of key-value pairs per page.
    :return: A list of dictionaries, each containing at most `page_size` key-value pairs.
    """
    keys = list(large_dict.keys())  # Extract keys to maintain order
    return [
        {key: large_dict[key] for key in keys[i : i + page_size]}
        for i in range(0, len(keys), page_size)
    ]


class Turn:
    """Turn"""

    def __init__(self, query: str, query_analysis: dict):
        self.query = query
        self.query_analysis = query_analysis
        self._embedded_results = None
        self._summary_raw = None
        self._summaries = None
        self._prompt = None
        self._response = None
        self._result_count = 0
        # target_type describes what is stored in the paged_document_map
        self.target_type: str = query_analysis["target"]
        self._paged_document_map: List[Dict] = None
        self.current_page = 0

    @property
    def embedded_results(self):
        """embedded_results"""
        return self._embedded_results

    @embedded_results.setter
    def embedded_results(self, er):
        self._embedded_results = er

    @property
    def summary_raw(self):
        """summary_raw"""
        return self._summary_raw

    @summary_raw.setter
    def summary_raw(self, sr):
        self._summary_raw = sr

    @property
    def summaries(self):
        """summaries"""
        return self._summaries

    @summaries.setter
    def summaries(self, s):
        self._summaries = s

    def do_synthesize(
        self, include_summaries: bool = True, include_uri: bool = True
    ) -> str:
        """syntehsize"""
        context = ""
        result_count = len(self.embedded_results["ids"][0])
        context = (
            f"{self.query_analysis["quantity"]} story[s] "
            f"from section: {self.query_analysis["section"]}\n"
        )
        for i in range(result_count):
            context += f"document {i}: {self.embedded_results['ids'][0][i]}\n"
            if include_uri:
                context += (
                    f"uri {i}: {self.embedded_results['metadatas'][0][i]['uri']}\n"
                )
            for i in range(3):
                title = self.embedded_results["metadatas"][0][i][f"headline{i}"]
                if title != "N/A":
                    context += "title: {title}\n"
            if include_summaries:
                context += (
                    f"summary: {self.summaries[self.embedded_results['ids'][0][i]]}\n"
                )

        context += (
            "" if self.response is None else f"Agent Last Response: {self.response}\n"
        )

        self._prompt = (
            f"Context: {{\n{context}\n}}\nUser Question: {self.query}\nAgent Answer:"
        )

        if len(self._prompt) > 2048 and include_summaries:
            print("\nremoving summaries from context\n")
            self.do_synthesize(include_summaries=False)
        if len(self._prompt) > 2048 and include_uri:
            print("\nremoving uris from context\n")
            self.do_synthesize(include_summaries=False, include_uri=False)

    def do_read(
        self,
        ordinal: int = 0,
        target: str = targets.DOCUMENT,
        archivedb: ChromaDBClient = None,
    ) -> str:
        """
        do_read
            an ordinal "n" + READ action infers reading n'th target from the current page

        param: ordinal: int
        param: target: str
        """
        text = ""
        if ordinal > 0:
            doc_map_idx = ordinal - 1
        else:
            # we dont know what the user is referring to
            doc_map_idx = -1
        match target:
            case targets.DOCUMENT:
                uri = list(
                    self._paged_document_map[self.current_page][doc_map_idx].keys()
                )[0]
                with open(uri, "r", encoding="utf8") as fh:
                    self.response = fh.read()
                    return
            case targets.SUMMARY:
                # if the context target is not summary, retrieve the summary for the corresponding
                # document
                uri = list(
                    self._paged_document_map[self.current_page][doc_map_idx].keys()
                )[0]
                if self.target_type != target:
                    self.response = archivedb.collection.get(ids=[], where={})
                    return
                else:
                    pass
            case _:
                pass

        return text

    def _prep_headline_maps(self):
        result_count = len(self.embedded_results["ids"][0])
        clean_map = {}
        _document_map = {
            i: {
                self.embedded_results["metadatas"][0][i]["uri"]: [
                    self.embedded_results["metadatas"][0][i][f"headline{hidx}"]
                    for hidx in range(3)
                    if self.embedded_results["metadatas"][0][i][f"headline{hidx}"]
                    != "N/A"
                ]
            }
            for i in range(result_count)
        }
        _document_map = {
            k: {u: [h.strip("'") for h in hs] for u, hs in v.items()}
            for k, v in _document_map.items()
        }
        seen = []
        i = 0
        for k, v in _document_map.items():
            for u, hs in v.items():
                # sort headlines by length
                hs.sort(key=lambda x: len(x))
                # only keep the shortest headline
                # only keep one reference to the story
                if hs[0] not in seen:
                    clean_map[i] = {u: hs[0]}
                    seen.append(hs[0])
                    i += 1

        self._result_count = len(clean_map)
        self._paged_document_map = paginate_dict(clean_map, 5)

    def do_list_headlines(self) -> str:
        """list_headlines"""
        context = ""
        if self._paged_document_map is None or len(self._paged_document_map) == 0:
            self._prep_headline_maps()

        # get the current page (default is first page)
        page = self._paged_document_map[self.current_page]
        context = (
            f"[{self.query_analysis["quantity"]} story[s] "
            f"from section: `{self.query_analysis["section"]}` "
            f"gives {self._result_count} documents. "
            f"Showing {len(page)} from page {self.current_page + 1}/"
            f"{len(self._paged_document_map)}.]\n"
        )
        for uri_map, document in page.items():
            # the document is a map of uri to shortest headline
            headline = list(document.values())[0]
            context += f"[headline {uri_map}: {headline}]\n"
        context += (
            "" if self.response is None else f"Previous Response: {self.response}\n"
        )

        self._prompt = f"Context:\n{context}\nUser Question: {self.query}"

    def prompt(self) -> str:
        """prompt"""
        action: str = self.query_analysis["action"]
        match action:
            case actions.READ:
                self.do_read(True, True)
            case actions.SUMMARIZE:
                self.do_synthesize(True, True)
            case actions.SYNTHESIZE:
                self.do_synthesize(True, True)
            case actions.LIST:
                self.do_list_headlines()

        return self._prompt

    @property
    def response(self):
        """response"""
        return self._response

    @response.setter
    def response(self, r):
        self._response = r
