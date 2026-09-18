# Nanocraft

![Nanocraft toolkit](docs/images/nanocraft_logo.png)

Nanocraft (Nanocrystal Assembly and Functionalization Toolkit) is an open-source Python toolkit for the generation, passivation, and functionalization of nanocrystals (NCs) and nanoparticles (NPs), such as quantum dots.

Operating locally through an accessible Jupyter Notebook interface, Nanocraft does not require advanced coding skills and enables the rapid generation and manipulation of large structural datasets by varying core stoichiometries, shapes, ligand types, and surface coverage.

## Key Features

- **Structure generation**: build bare nanocrystals from crystallographic CIF files or pre-built XYZ supercells using geometric cutting masks (spheres, cubes, octahedra, truncated octahedra, cylinders, and more).
- **Automated surface passivation**: charge-based passivation scheme relying on atomic oxidation states and coordination numbers, ensuring fully saturated, charge-neutral structures.
- **Facet-selective functionalization**: attach ligands or capping fragments to specific crystallographic facets.
- **Ligand substitution and exchange**: replace existing capping groups with new ligands to model mixed-ligand shells or ligand exchange.
- **Molecular editing**: introduce atomic substitutions, vacancies, alloys, and controlled geometric disorder (shuffling).
- **Batch processing**: explore large parameter grids and run generation/functionalization tasks in parallel via Python multiprocessing.
- **Interoperability**: reads and writes exclusively in the standard XYZ format, for direct use in molecular dynamics, DFT, and machine-learning workflows.

## Getting Started

Nanocraft requires **Python 3.10+** and has no specific operating system requirements. See [`INSTALL`](INSTALL) for detailed installation instructions.

Once installed, open the main Jupyter notebook provided in the repository to start generating and functionalizing nanostructures, no scripting is required. 

A dedicated visualization notebook is also included to inspect generated or imported structures.

## Documentation

A complete User Guide describing all functionalities and their arguments is provided in this repository. It includes a description of configuration files used by the toolkit and how to modify them.

## Example Applications

Nanocraft has been used to generate extensive structural datasets for a variety of systems, including CdSe and CdTe (zincblende and wurtzite), GaN and InGaN (wurtzite), TiO2 (anatase), Pt, and Au (fcc). Structures, as well as inputs, are provided in this repository as a .tar.gz archive.

## Citing Nanocraft

If you use Nanocraft in your research, please cite the associated publication (see `CITATION` file or repository description for details).

## License

Nanocraft is released under the [MIT License](LICENSE).

## Acknowledgments

This work was supported by the CNRS MITI program (INANOMEP International Research Project) and a CNRS public doctoral contract awarded to Justin Grill, main contributor to this project.
