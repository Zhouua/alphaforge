import ast
import json
import tempfile
import unittest
from pathlib import Path

from factor_library_io import write_factor_library
from experiment_protocol import (
    BACKTEST,
    FEATURES,
    MODEL,
    QLIB_TARGET,
    SPLITS,
    TARGET_HORIZON,
    protocol_dict,
)
class AlignedProtocolTest(unittest.TestCase):
    def test_exact_protocol(self):
        self.assertEqual(
            (
                SPLITS.train_start,
                SPLITS.train_end,
                SPLITS.valid_start,
                SPLITS.valid_end,
                SPLITS.test_start,
                SPLITS.test_end,
            ),
            (
                "2010-01-01",
                "2019-11-30",
                "2020-01-01",
                "2021-11-30",
                "2022-01-01",
                "2025-12-31",
            ),
        )
        self.assertEqual(FEATURES, ("open", "high", "low", "close", "volume", "vwap"))
        self.assertEqual(QLIB_TARGET, "Ref($open, -11) / Ref($open, -1) - 1")
        self.assertEqual(TARGET_HORIZON, 10)
        self.assertEqual((MODEL.estimator, MODEL.alpha), ("ridge", 10.0))
        self.assertFalse(MODEL.fit_intercept)
        self.assertEqual((BACKTEST.topk, BACKTEST.n_drop), (50, 5))
        self.assertEqual(BACKTEST.deal_price, "open")
        self.assertEqual(BACKTEST.account, 100_000_000)
        self.assertEqual(
            (BACKTEST.open_cost, BACKTEST.close_cost, BACKTEST.min_cost),
            (0.0005, 0.0015, 5.0),
        )

    def test_search_loader_has_no_test_interface(self):
        root = ast.parse(
            Path("gan/utils/data.py").read_text(encoding="utf-8")
        )
        function = next(
            node
            for node in root.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_search_data_by_dates"
        )
        parameters = [argument.arg for argument in function.args.args]
        self.assertNotIn("test_start", parameters)
        self.assertNotIn("test_end", parameters)

    def test_frozen_metadata_omits_test_dates(self):
        splits = protocol_dict(include_test=False)["splits"]
        self.assertNotIn("test_start", splits)
        self.assertNotIn("test_end", splits)

    def test_public_evaluator_does_not_fit(self):
        source = Path("public_test.py").read_text(encoding="utf-8")
        self.assertNotIn(".fit(", source)
        self.assertNotIn("factor_library", source)

    def test_gp_and_dso_generation_use_search_loader_only(self):
        for filename in ("train_GP.py", "train_DSO.py"):
            tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
            imported_names = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "gan.utils.data"
                for alias in node.names
            }
            self.assertEqual(imported_names, {"get_search_data_by_dates"})
            source = Path(filename).read_text(encoding="utf-8")
            self.assertNotIn("pred_test", source)

    def test_common_library_requires_fifty_train_factors(self):
        factors = [(f"factor_{index}", index / 1000) for index in range(60)]
        with tempfile.TemporaryDirectory() as directory:
            library_path = write_factor_library(
                factors,
                directory,
                method="GP",
                method_id="gp",
                seed=0,
                run_name="unit",
                library_size=50,
                min_factors=50,
            )
            library = json.loads(library_path.read_text(encoding="utf-8"))
            metadata = json.loads(
                (Path(directory) / "run_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(library["exprs"]), 50)
            self.assertEqual(library["selection_data"], "train")
            self.assertFalse(metadata["test_data_loaded"])

    def test_vendored_dso_uses_python311_collection_apis(self):
        source = Path(
            "third_party/dso_pytorch/dso/dso/utils.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("collections.Mapping", source)
        self.assertIn("from collections.abc import Mapping", source)


if __name__ == "__main__":
    unittest.main()
