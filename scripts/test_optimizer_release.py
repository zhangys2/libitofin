"""Fork contract: itofin-optimize ships the workspace license and is not published here."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OptimizerReleaseTests(unittest.TestCase):
    def test_optimizer_bundles_unmodified_workspace_bsd_license(self):
        root_license = (ROOT / "LICENSE").read_bytes()
        crate = ROOT / "crates/itofin-optimize"
        self.assertEqual((crate / "LICENSE").read_bytes(), root_license)
        package = tomllib.loads((crate / "Cargo.toml").read_text())["package"]
        self.assertEqual(package["license"], "BSD-3-Clause")
        self.assertTrue(package["version"]["workspace"])
        self.assertNotIn("include", package)
        self.assertNotIn("exclude", package)

    def test_this_fork_does_not_publish_through_semantic_release(self):
        self.assertFalse((ROOT / ".github/workflows/semantic-release.yml").is_file())
        readme = (ROOT / "crates/itofin-optimize/README.md").read_text()
        self.assertIn("does not publish `itofin-optimize` through `semantic-release.yml`", readme)
        self.assertIn("including 0.37.0", readme)
        self.assertNotIn("Owner `benbenbang`", readme)
        self.assertNotIn("trusted-publisher configuration are verified", readme)


if __name__ == "__main__":
    unittest.main()
