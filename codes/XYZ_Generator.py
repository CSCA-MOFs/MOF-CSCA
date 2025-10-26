from pathlib import Path
import pymatgen.core as pmcore

CIF_DIR = Path("path_to_cifs")
XYZ_DIR = Path("XYZ_Generated")
SUPERCELL = 64.0

def choose_supercell_multipliers(a, b, c, target):
    def mult(x):
        if x <= 0:
            return 1
        m = round(target / x)
        return max(1, m)
    return mult(a), mult(b), mult(c)

def cif_to_xyz(cif_path, xyz_path, supercell=64.0):
    try:
        struct = pmcore.Structure.from_file(str(cif_path), primitive=False)
        a, b, c = struct.lattice.abc
        n_a, n_b, n_c = choose_supercell_multipliers(a, b, c, supercell)
        struct.make_supercell([n_a, n_b, n_c])
        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        with xyz_path.open("w", encoding="utf-8") as f:
            f.write(f"{len(struct)}\n")
            f.write("Supercell {} {} {} | Lattice {:.6f} {:.6f} {:.6f}\n".format(
                n_a, n_b, n_c, struct.lattice.a, struct.lattice.b, struct.lattice.c
            ))
            for site in struct:
                x, y, z = site.coords
                f.write(f"{site.species_string} {x:.8f} {y:.8f} {z:.8f}\n")
        return True
    except Exception:
        return False

def main():
    XYZ_DIR.mkdir(parents=True, exist_ok=True)
    if not CIF_DIR.is_dir():
        return
    cif_files = sorted([p for p in CIF_DIR.iterdir() if p.suffix.lower() == ".cif"])
    if not cif_files:
        return
    for cif_path in cif_files:
        xyz_path = XYZ_DIR / (cif_path.stem + ".xyz")
        cif_to_xyz(cif_path, xyz_path, SUPERCELL)

if __name__ == "__main__":
    main()
