"""Bead identity in PrISM's IHM path. Inputs come from tests/data/make_fixtures.py.

Collected by pytest, and runnable without it (no ihmv image ships pytest):

	python3 -m unittest tests.test_ihm_identity -v        # from the repo root
"""
import os
import sys
import unittest

import ihm.reader
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ihm_parser  # noqa: E402
import main as prism_main  # noqa: E402

DATA = os.path.join(HERE, "data")


def model_group(name):
	with open(os.path.join(DATA, name)) as f:
		system = ihm.reader.read(f)[0]
	return system.ensembles[0].model_group


def model0_xyz(group, asym_id, seq_begin, seq_end, kind):
	"""Find a bead in model 0 by identity, independently of PrISM's walk order."""
	model = list(group)[0]
	if kind == "sphere":
		hits = [s for s in model._spheres
			if s.asym_unit._id == asym_id and tuple(s.seq_id_range) == (seq_begin, seq_end)]
	else:
		hits = [a for a in model._atoms
			if a.asym_unit._id == asym_id and a.seq_id == seq_begin and a.atom_id == "CA"]
	assert len(hits) == 1, (asym_id, seq_begin, seq_end, kind, len(hits))
	return [hits[0].x, hits[0].y, hits[0].z]


class TestGetAllAttributesIhm(unittest.TestCase):
	def check(self, name, kinds):
		group = model_group(name)
		coords, radius, mass, ps_names, bead_ids = ihm_parser.get_all_attributes_ihm(group, with_ids=True)
		n_models, n_beads = coords.shape[0], coords.shape[1]
		self.assertEqual(len(bead_ids), n_beads)
		self.assertEqual({b[3] for b in bead_ids}, kinds)
		# with_ids adds bead_ids and changes nothing else
		default = ihm_parser.get_all_attributes_ihm(group)
		for got, want in zip([coords, radius, mass], default[:3]):
			np.testing.assert_array_equal(got, want)
		self.assertEqual(ps_names, default[3])
		for i, (asym_id, begin, end, kind) in enumerate(bead_ids):
			if kind == "atom":
				self.assertEqual(begin, end)
			np.testing.assert_allclose(coords[0][i], model0_xyz(group, asym_id, begin, end, kind))

	def test_coarse(self):
		self.check("coarse_9a3s.cif", {"sphere"})

	def test_atomic(self):
		self.check("atomic_9a1s.cif", {"atom"})

	def test_mixed(self):
		self.check("mixed_9a1s.cif", {"sphere", "atom"})

	def test_default_keeps_four_values(self):
		# IHMValidation < 3.3 unpacks exactly four values
		self.assertEqual(len(ihm_parser.get_all_attributes_ihm(model_group("atomic_9a1s.cif"))), 4)


class TestRunPrismIhm(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.args = ihm_parser.get_all_attributes_ihm(model_group("mixed_9a1s.cif"), with_ids=True)

	def test_identity_columns(self):
		coords, radius, mass, ps_names, bead_ids = self.args
		df = prism_main.run_prism_ihm(coords, mass, radius, ps_names, classes=3, bead_ids=bead_ids)
		self.assertEqual(list(df["asym_id"]), [b[0] for b in bead_ids])
		self.assertEqual(list(df["seq_id_begin"]), [b[1] for b in bead_ids])
		self.assertEqual(list(df["seq_id_end"]), [b[2] for b in bead_ids])
		self.assertEqual(list(df["kind"]), [b[3] for b in bead_ids])
		self.assertEqual(df["seq_id_begin"].dtype.kind, "i")
		self.assertEqual(df["seq_id_end"].dtype.kind, "i")

	def test_columns_unchanged_without_bead_ids(self):
		coords, radius, mass, ps_names, _ = self.args
		df = prism_main.run_prism_ihm(coords, mass, radius, ps_names, classes=3)
		self.assertEqual(list(df.columns),
			["Bead", "Bead Name", "Type", "Class", "Patch", "x", "y", "z", "r"])

	def test_bead_ids_only_add_columns(self):
		coords, radius, mass, ps_names, bead_ids = self.args
		plain = prism_main.run_prism_ihm(coords, mass, radius, ps_names, classes=3)
		with_ids = prism_main.run_prism_ihm(coords, mass, radius, ps_names, classes=3, bead_ids=bead_ids)
		pd.testing.assert_frame_equal(
			with_ids.drop(columns=["asym_id", "seq_id_begin", "seq_id_end", "kind"]), plain)


if __name__ == "__main__":
	unittest.main()
