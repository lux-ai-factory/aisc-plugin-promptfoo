"""Generic byte->structure decoders for user-uploaded datasets.

These are format decoders ONLY — the core stack passes raw bytes, the provider
turns them into a structure, and the plugin owns all semantic validation/parsing.
Mirrors the strongreject/ragas plugins' providers for cross-plugin consistency.

For promptfoo eval the user uploads a test CSV that promptfoo parses natively, so
the plugin passes the bytes straight through (RawBytesProvider) and writes them to
``tests.csv`` in the run workspace. CsvToPandasProvider is kept available for any
future plugin-side inspection of the CSV.
"""
import io

import pandas as pd
from pandas import DataFrame

from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider


class RawBytesProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> bytes:
        return file_content


class CsvToPandasProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> DataFrame:
        return pd.read_csv(io.BytesIO(file_content))
