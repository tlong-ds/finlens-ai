import math
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from src import retrieval
from src.embeddings import (
    EMBEDDING_MODEL_DEFAULT,
    EMBEDDING_REVISION_DEFAULT,
    EMBEDDING_VECTOR_SIZE,
    DenseEmbeddingModel,
)


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.kwargs: dict[str, object] = {}

    def encode(self, inputs: list[str], **kwargs: object) -> list[list[float]]:
        self.inputs = inputs
        self.kwargs = kwargs
        return [[2.0] + [0.0] * (EMBEDDING_VECTOR_SIZE - 1) for _ in inputs]


class _SlowFakeEncoder:
    def __init__(self) -> None:
        self.load_count = 0

    def load(self) -> None:
        self.load_count += 1
        time.sleep(0.05)


class DenseEmbeddingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = DenseEmbeddingModel()
        self.fake_model = _FakeSentenceTransformer()
        self.encoder._model = self.fake_model

    def test_passages_are_unprefixed_and_normalized(self) -> None:
        vectors = self.encoder.encode_passages(["Loại bảng: Báo cáo kết quả kinh doanh"])

        self.assertEqual(
            self.fake_model.inputs,
            ["Loại bảng: Báo cáo kết quả kinh doanh"],
        )
        self.assertTrue(self.fake_model.kwargs["normalize_embeddings"])
        self.assertEqual(len(vectors[0]), EMBEDDING_VECTOR_SIZE)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vectors[0])), 1.0)

    def test_queries_are_unprefixed(self) -> None:
        self.encoder.encode_queries(["Doanh thu thuần của VJC năm 2024?"])

        self.assertEqual(
            self.fake_model.inputs,
            ["Doanh thu thuần của VJC năm 2024?"],
        )

    def test_defaults_pin_granite_multilingual_r2(self) -> None:
        self.assertEqual(
            EMBEDDING_MODEL_DEFAULT,
            "ibm-granite/granite-embedding-97m-multilingual-r2",
        )
        self.assertEqual(
            EMBEDDING_REVISION_DEFAULT,
            "835ad14087e140460703cf0fae09f97d469d65c2",
        )

    def test_empty_inputs_do_not_load_or_call_model(self) -> None:
        self.assertEqual(self.encoder.encode_passages([]), [])
        self.assertEqual(self.fake_model.inputs, [])

    def test_concurrent_first_access_loads_one_shared_encoder(self) -> None:
        fake_encoder = _SlowFakeEncoder()
        with (
            patch("src.retrieval._embedding_model", None),
            patch(
                "src.retrieval.DenseEmbeddingModel.from_env",
                return_value=fake_encoder,
            ) as factory,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            models = list(executor.map(lambda _: retrieval._get_embedding_model(), range(2)))

        self.assertIs(models[0], models[1])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(fake_encoder.load_count, 1)


if __name__ == "__main__":
    unittest.main()
