"""DeepSeek-V4.1-Flash on an Apple Silicon Mac: the fit numbers (`strata_control/dsv41_fit.py`, `mac_expert_cap_gb` and
`mac_plan`).  The cap mirrors the engine's own rule, so 128 GiB must give the 71.4 GB of VRAM experts measured on the M5
Max on 2026-10-25.  Pure: nothing here builds, downloads or starts anything."""

from __future__ import annotations

import unittest

from strata_control import dsv41_fit as F


class MacExpertCap(unittest.TestCase):
    def test_the_m5_max_gives_the_measured_71_4_gb(self):
        self.assertAlmostEqual(F.mac_expert_cap_gb(128), 71.4, delta=0.3)

    def test_a_32_gb_mac_mini_keeps_little(self):
        # 32 GiB: leave = 15% (4.8 GiB = 5.15 GB) and the working set caps it at 3/4 of 34.4 GB
        cap = F.mac_expert_cap_gb(32)
        self.assertGreater(cap, 0.0)
        self.assertLess(cap, 131.0 * 0.25)          # well under a quarter of the 2.00 pack
        self.assertLess(F.mac_expert_cap_gb(16), F.mac_expert_cap_gb(32))
        self.assertEqual(F.mac_expert_cap_gb(0), 0.0)

    def test_a_big_mac_stops_at_the_working_set(self):
        # 512 GiB: the file-cache floor rule leaves 40% and the device share is the tighter of the two? no: 40% wins
        cap = F.mac_expert_cap_gb(512)
        self.assertGreater(cap, 300.0)              # far more than any pack needs


class MacPlan(unittest.TestCase):
    def test_the_default_pack_on_the_m5_max_streams(self):
        p = F.mac_plan(128, "IQ2_XS")
        self.assertEqual(p["verdict"], "streams")
        self.assertAlmostEqual(p["resident_gb"], 71.4, delta=0.3)
        self.assertAlmostEqual(p["streaming_gb"], 149.06 - 71.4, delta=0.3)
        self.assertEqual(p["dense_gb"], 6.17)
        self.assertIn("streams", p["words"])
        self.assertIn("M5 Max", p["words"])

    def test_the_1_75_pack_streams_less_on_the_same_mac(self):
        small = F.mac_plan(128, "IQ2_XS")["streaming_gb"]
        self.assertLess(F.mac_plan(128, "IQ1_M")["streaming_gb"], small)
        self.assertGreater(F.mac_plan(128, "IQ2_XXS")["streaming_gb"], F.mac_plan(128, "IQ1_M")["streaming_gb"])

    def test_a_mac_with_enough_memory_fits(self):
        p = F.mac_plan(512, "IQ2_XS")
        self.assertEqual(p["verdict"], "fits")
        self.assertEqual(p["streaming_gb"], 0.0)

    def test_a_small_mac_is_told_plainly(self):
        p = F.mac_plan(32, "IQ2_XS")
        self.assertEqual(p["verdict"], "small")
        self.assertTrue(p["warnings"])
        self.assertTrue(any("read from the SSD" in w or "Expert" in w or "quarter" in w for w in p["warnings"]))

    def test_disk_and_unknown_pack(self):
        p = F.mac_plan(128, "IQ2_XS", disk_free_gb=50.0)
        self.assertFalse(p["disk_ok"])
        self.assertTrue(any("disk space" in w for w in p["warnings"]))
        self.assertTrue(F.mac_plan(128, "IQ2_XS", disk_free_gb=900.0)["disk_ok"])
        with self.assertRaises(ValueError):
            F.mac_plan(128, "IQ9_Z")

    def test_every_pack_is_described(self):
        for q in F.MAC_PACKS:
            p = F.mac_plan(128, q)
            self.assertTrue(p["words"])
            self.assertGreater(p["disk_need_gb"], F.MAC_ENGRAM_GB)


if __name__ == "__main__":
    unittest.main()
