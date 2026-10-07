"""Build the small IHM fixtures used by tests/test_ihm_identity.py.

	python3 tests/data/make_fixtures.py        # from the repo root

Downloads PDB-IHM entries 9A3S and 9A1S and writes three files next to this
script, each keeping the first 3 models of the first ensemble:

	coarse_9a3s.cif   9A3S, coarse-grained only (one-residue spheres)
	atomic_9a1s.cif   9A1S, atomic only
	mixed_9a1s.cif    9A1S with residues up to the chain midpoint rewritten as
	                  one-residue spheres at their CA. Synthetic: no released
	                  entry mixes atoms and spheres in one model yet, and the
	                  parser has to handle it when one does.
"""
import io
import os
import urllib.request

import ihm.dumper
import ihm.model
import ihm.reader
import ihm.representation

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://pdb-ihm.org/cif/{}.cif"
N_MODELS = 3


def fetch(entry):
	with urllib.request.urlopen(URL.format(entry)) as r:
		return ihm.reader.read(io.StringIO(r.read().decode()))[0]


def trim(system, n=N_MODELS):
	"""Keep the first n models of the first ensemble's model group.

	Restraints record a fit per model (9A3S's 3DEM restraint lists all 11), and
	the dumper cannot write fits for models that are in no group any more, so
	drop those as well.
	"""
	group = system.ensembles[0].model_group
	models = list(group)[:n]
	group[:] = models
	kept = {id(m) for m in models}
	for restraint in system.restraints:
		fit_maps = [getattr(restraint, "fits", None)]
		fit_maps += [getattr(xl, "fits", None) for xl in getattr(restraint, "cross_links", None) or []]
		for fits in fit_maps:
			if isinstance(fits, dict):
				for model in [m for m in fits if id(m) not in kept]:
					del fits[model]
	return models


def write(system, name):
	with open(os.path.join(HERE, name), "w") as f:
		ihm.dumper.write(f, [system])


def make_mixed(system):
	"""Rewrite the first half of the (single) chain as one-residue spheres."""
	models = trim(system)
	representation = models[0].representation
	asym = representation[0].asym_unit.asym
	seq_ids = sorted({a.seq_id for a in models[0]._atoms})
	mid = seq_ids[len(seq_ids) // 2]
	for model in models:
		atoms, spheres = [], []
		for atom in model._atoms:
			if atom.seq_id > mid:
				atoms.append(atom)
			elif atom.atom_id == "CA":
				spheres.append(ihm.model.Sphere(
					asym_unit=atom.asym_unit, seq_id_range=(atom.seq_id, atom.seq_id),
					x=atom.x, y=atom.y, z=atom.z, radius=2.7))
		model._atoms, model._spheres = atoms, spheres
	lo, hi = asym.seq_id_range
	representation[:] = [
		ihm.representation.ResidueSegment(asym(lo, mid), rigid=False, primitive="sphere"),
		ihm.representation.AtomicSegment(asym(mid + 1, hi), rigid=False)]
	return system


def main():
	coarse = fetch("9a3s")
	trim(coarse)
	write(coarse, "coarse_9a3s.cif")

	atomic = fetch("9a1s")
	trim(atomic)
	write(atomic, "atomic_9a1s.cif")

	write(make_mixed(fetch("9a1s")), "mixed_9a1s.cif")


if __name__ == "__main__":
	main()
