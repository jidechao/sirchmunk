# Copyright (c) ModelScope Contributors. All rights reserved.
"""Integration tests for AgenticSearch.search() entry point.

Every test calls the real search() with a real LLM and real files —
no mocks, no patches.  Configuration is loaded exclusively from
tests/.env.test.

Return-value contract:
    - Default (return_context=False):
        - FAST / DEEP text search → ``str``
        - FILENAME_ONLY           → ``List[Dict]``
    - return_context=True:
        - All modes → ``SearchContext``
            .answer  : str
            .cluster : KnowledgeCluster | None
"""

import asyncio
import json
import os
import unittest
from pathlib import Path
from typing import Dict, List

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.schema.knowledge import KnowledgeCluster
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.search import AgenticSearch


# ------------------------------------------------------------------ #
# Test configuration — loaded from .env.test
# ------------------------------------------------------------------ #

_TESTS_DIR = Path(__file__).resolve().parent
_ENV_FILE = _TESTS_DIR / ".env.test"


def _load_env(path: Path) -> Dict[str, str]:
    """Parse a dotenv-style file into a dict (no shell expansion)."""
    cfg: Dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Test env file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            cfg[key.strip()] = value.strip()
    return cfg


_CFG = _load_env(_ENV_FILE)


def _cfg(key: str, default: str = "") -> str:
    return _CFG.get(key, default)


def _cfg_int(key: str, default: int = 0) -> int:
    return int(_CFG.get(key, str(default)))


def _cfg_float(key: str, default: float = 0.0) -> float:
    return float(_CFG.get(key, str(default)))


def _cfg_bool(key: str, default: bool = False) -> bool:
    return _CFG.get(key, str(default)).lower() in ("true", "1", "yes")


def _cfg_list(key: str) -> List[str]:
    raw = _cfg(key)
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else []


# ------------------------------------------------------------------ #
# Base test class — real AgenticSearch, real LLM, real files
# ------------------------------------------------------------------ #

class _BaseSearchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        api_key = _cfg("LLM_API_KEY")
        if not api_key:
            raise unittest.SkipTest("LLM_API_KEY not configured in .env.test")

        search_paths = _cfg_list("SEARCH_PATHS")
        if not search_paths:
            raise unittest.SkipTest("SEARCH_PATHS not configured in .env.test")

        cls.search_paths = search_paths

        llm = OpenAIChat(
            base_url=_cfg("LLM_BASE_URL"),
            api_key=api_key,
            model=_cfg("LLM_MODEL_NAME"),
            timeout=_cfg_float("LLM_TIMEOUT", 60.0),
        )

        work_path = _cfg("SIRCHMUNK_WORK_PATH") or os.path.join(
            os.path.expanduser("~"), ".sirchmunk", "test_work",
        )

        cls.searcher = AgenticSearch(
            llm=llm,
            work_path=work_path,
            paths=search_paths,
            verbose=_cfg_bool("SIRCHMUNK_VERBOSE"),
            reuse_knowledge=_cfg_bool("SIRCHMUNK_ENABLE_CLUSTER_REUSE"),
        )

    def _run(self, coro):
        return asyncio.run(coro)


# ================================================================== #
#  FAST MODE                                                           #
# ================================================================== #

class TestSearchFastMode(_BaseSearchTest):

    def test_fast_returns_answer_string(self):
        """FAST mode returns a non-empty string answer."""
        query = _cfg("TEST_QUERY_FAST", "transformer attention mechanism")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FAST",
        ))

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_fast_return_context(self):
        """FAST + return_context returns a SearchContext with answer and cluster."""
        query = _cfg("TEST_QUERY_FAST", "transformer attention mechanism")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FAST",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        self.assertIsInstance(result.answer, str)
        self.assertTrue(len(result.answer) > 0)
        self.assertIsNotNone(result.cluster)
        self.assertIsInstance(result.cluster, KnowledgeCluster)
        self.assertTrue(result.cluster.id.startswith("FS"))
        self.assertEqual(result.answer, result.cluster.content)

    def test_fast_context_serializable(self):
        """SearchContext.to_dict() produces a JSON-serializable dict."""
        query = _cfg("TEST_QUERY_FAST", "transformer attention mechanism")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FAST",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("answer", d)
        serialized = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(serialized, str)


# ================================================================== #
#  DEEP MODE                                                           #
# ================================================================== #

class TestSearchDeepMode(_BaseSearchTest):

    def test_deep_returns_answer_string(self):
        """DEEP mode returns a non-empty string answer."""
        query = _cfg("TEST_QUERY_DEEP", "How does transformer attention work?")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="DEEP",
        ))

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_deep_return_context(self):
        """DEEP + return_context returns a SearchContext with answer and cluster."""
        query = _cfg("TEST_QUERY_DEEP", "How does transformer attention work?")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="DEEP",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        self.assertIsInstance(result.answer, str)
        self.assertTrue(len(result.answer) > 0)
        self.assertIsNotNone(result.cluster)
        self.assertIsInstance(result.cluster, KnowledgeCluster)
        self.assertEqual(result.answer, result.cluster.content)


# ================================================================== #
#  FILENAME_ONLY MODE                                                  #
# ================================================================== #

class TestSearchFilenameOnly(_BaseSearchTest):

    def test_filename_only_returns_list(self):
        """FILENAME_ONLY returns a list of file match dicts."""
        query = _cfg("TEST_QUERY_FILENAME", "notes")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FILENAME_ONLY",
        ))

        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_filename_only_no_matches(self):
        """No matches returns an error string."""
        result = self._run(self.searcher.search(
            query="__nonexistent_file_xyz_42__",
            paths=self.search_paths,
            mode="FILENAME_ONLY",
        ))

        self.assertIsInstance(result, str)
        self.assertIn("No files found", result)


# ================================================================== #
#  PATH VALIDATION                                                     #
# ================================================================== #

class TestPathValidation(unittest.TestCase):
    """Unit tests for AgenticSearch.validate_search_paths (no LLM)."""

    def test_rejects_hyphen_prefix(self):
        clean = AgenticSearch.validate_search_paths(["--help", "/tmp"])
        self.assertEqual(len(clean), 1)
        self.assertNotIn("--help", clean)

    def test_rejects_null_byte(self):
        clean = AgenticSearch.validate_search_paths(["/tmp/foo\x00bar"])
        self.assertEqual(clean, [])

    def test_rejects_nonexistent_with_require_exists(self):
        clean = AgenticSearch.validate_search_paths(
            ["/absolutely/does/not/exist/xyz"],
            require_exists=True,
        )
        self.assertEqual(clean, [])

    def test_accepts_valid_url(self):
        clean = AgenticSearch.validate_search_paths(
            ["https://example.com/docs"],
        )
        self.assertEqual(clean, ["https://example.com/docs"])

    def test_rejects_malformed_url(self):
        clean = AgenticSearch.validate_search_paths(["https://"])
        self.assertEqual(clean, [])

    def test_deduplicates(self):
        clean = AgenticSearch.validate_search_paths(["/tmp", "/tmp", "/tmp"])
        self.assertEqual(len(clean), 1)


if __name__ == "__main__":
    unittest.main()
