"""Load cytometry samples from HDF5 files."""

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

FEATURE_SYNONYMS = {
    "CD8a": "CD8",
    "CD8b": "CD8",
    "PD1": "PD-1",
    "PDL1": "PD-L1",
    "PD-L1": "PD-L1",
    "HLADR": "HLA-DR",
    "HLA_DR": "HLA-DR",
    "GzmB": "GranzymeB",
    "GZMB": "GranzymeB",
    "KI67": "Ki-67",
    "Ki67": "Ki-67",
    "CTLA4": "CTLA-4",
    "gdTCR": "TCRgd",
    "PANGT": "TCRgd",
    "FOXP3": "FOXP3",
    "Tbet": "T-bet",
    "IFNG": "IFN-g",
    "TNF": "TNF-a",
    "IL2": "IL-2",
    "IL17": "IL-17",
    "CCR7": "CD197",
    "GranzymeB": "GranzymeB",
    "TCR_Va24-Ja18": "Va24-Ja18",
    "FcER1": "FceR1",
    "FceR1": "FceR1",
    "CXCR3": "CD183",
    "CXCR5": "CD185",
    "CCR4": "CD194",
    "CCR6": "CD196",
    "FceRI": "FceR1",
    "CRTH2": "CD294",
}


def canonicalize_marker(
    name: str,
) -> str:
    """Map a raw marker name to its canonical form."""
    return FEATURE_SYNONYMS.get(name, name)


class CyTOFDataset(Dataset):
    """Load cytometry samples with a canonical shared marker panel."""

    ZERO_THRESHOLD = 1e-6

    def __init__(
        self,
        data_dirs: str | Path | Sequence[str | Path],
        subset_size: int | None = None,
        cell_type_removal: Iterable[str] | None = None,
        marker_dirs: str | Path | Sequence[str | Path] | None = None,
    ) -> None:
        """Initialize the dataset and determine its shared marker panel."""
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        if isinstance(marker_dirs, (str, Path)):
            marker_dirs = [marker_dirs]

        self.data_dirs = [Path(data_directory) for data_directory in data_dirs]
        self.marker_dirs = [
            Path(marker_directory) for marker_directory in marker_dirs or []
        ]
        self.data_directories = self.data_dirs
        self.marker_directories = self.marker_dirs
        self.subset_size = subset_size
        self.cell_type_removal = set(cell_type_removal or [])
        self.file_paths, self.file_to_directory = self._get_file_paths()
        self.sample_names = list(self.file_paths)
        self.cell_type_mapping: dict[str, int] = {}
        self.reverse_mapping: dict[int, str] = {}
        self.discovered_types: set[str] = set()
        self.shared_markers, self.directory_column_indices = self._build_shared_panel()

        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                f"Initialized dataset: {len(self.sample_names)} samples from "
                f"{len(self.data_directories)} training directories"
            )
            if self.marker_directories:
                print(
                    f"Marker-only directories ({len(self.marker_directories)}): "
                    f"{self.marker_directories}"
                )
            print(f"Shared markers ({len(self.shared_markers)}): {self.shared_markers}")
            if self.cell_type_removal:
                print(f"Removing cell types: {sorted(self.cell_type_removal)}")

    def _get_file_paths(self) -> tuple[dict[str, Path], dict[str, Path]]:
        """Map sample names to files and source directories."""
        file_paths: dict[str, Path] = {}
        file_to_directory: dict[str, Path] = {}
        for data_directory in self.data_directories:
            for file_path in data_directory.iterdir():
                if file_path.suffix != ".h5":
                    continue
                sample_name = file_path.stem
                if sample_name in file_paths:
                    raise ValueError(
                        f"Duplicate sample name '{sample_name}' found in "
                        f"{data_directory}"
                    )
                file_paths[sample_name] = file_path
                file_to_directory[sample_name] = data_directory
        return file_paths, file_to_directory

    def _build_shared_panel(
        self,
    ) -> tuple[list[str], dict[Path, list[int]]]:
        """Build a canonical shared marker panel across all directories."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        canonical_panels: dict[Path, list[str]] = {}
        all_directories = self.data_directories + self.marker_directories

        for data_directory in all_directories:
            is_marker_only = data_directory in self.marker_directories
            for file_path in data_directory.iterdir():
                if file_path.suffix != ".h5":
                    continue
                with h5py.File(file_path, "r") as h5_file:
                    raw_markers = [
                        marker.decode("utf-8") for marker in h5_file["feature_names"][:]
                    ]
                canonical_markers = [
                    canonicalize_marker(marker) for marker in raw_markers
                ]
                canonical_panels[data_directory] = canonical_markers

                if local_rank == 0:
                    tag = " [marker-only]" if is_marker_only else ""
                    renamed_markers = [
                        (raw_name, canonical_name)
                        for raw_name, canonical_name in zip(
                            raw_markers,
                            canonical_markers,
                            strict=True,
                        )
                        if raw_name != canonical_name
                    ]
                    print(
                        f"[Panel] {data_directory.name}{tag}: "
                        f"{len(raw_markers)} markers, "
                        f"{len(renamed_markers)} renamed"
                    )
                    for raw_name, canonical_name in renamed_markers:
                        print(f"[Panel]   {raw_name} -> {canonical_name}")
                break

        if len(canonical_panels) != len(all_directories):
            missing_directories = [
                directory
                for directory in all_directories
                if directory not in canonical_panels
            ]
            raise FileNotFoundError(
                f"No HDF5 files found in directories: {missing_directories}"
            )

        shared_marker_set = set(canonical_panels[all_directories[0]])
        for canonical_panel in canonical_panels.values():
            shared_marker_set &= set(canonical_panel)
        shared_markers = sorted(shared_marker_set)

        directory_column_indices: dict[Path, list[int]] = {}
        for data_directory in self.data_directories:
            canonical_marker_indices: dict[str, int] = {}
            for index, canonical_marker in enumerate(canonical_panels[data_directory]):
                if canonical_marker not in canonical_marker_indices:
                    canonical_marker_indices[canonical_marker] = index
            directory_column_indices[data_directory] = [
                canonical_marker_indices[marker] for marker in shared_markers
            ]

        if local_rank == 0:
            print(f"[Panel] Shared markers ({len(shared_markers)}): {shared_markers}")
        return shared_markers, directory_column_indices

    def _register_cell_type(
        self,
        cell_type: str,
    ) -> None:
        """Register a cell type using deterministic alphabetical ordering."""
        self.discovered_types.add(cell_type)
        sorted_types = sorted(self.discovered_types)
        self.cell_type_mapping = {
            cell_type_name: index for index, cell_type_name in enumerate(sorted_types)
        }
        self.reverse_mapping = dict(enumerate(sorted_types))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sample_names)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Load and sample one cytometry specimen."""
        sample_name = self.sample_names[index]
        file_path = self.file_paths[sample_name]

        with h5py.File(file_path, "r") as h5_file:
            raw_features = torch.tensor(h5_file["data"][:])
            cell_type_names = [
                cell_type.decode("utf-8") for cell_type in h5_file["cell_types"][:]
            ]

        column_indices = self.directory_column_indices[
            self.file_to_directory[sample_name]
        ]
        features = raw_features[:, column_indices]

        if self.cell_type_removal:
            keep_mask = torch.tensor(
                [
                    cell_type not in self.cell_type_removal
                    for cell_type in cell_type_names
                ],
                dtype=torch.bool,
            )
            features = features[keep_mask]
            cell_type_names = [
                cell_type
                for cell_type in cell_type_names
                if cell_type not in self.cell_type_removal
            ]

        for cell_type in cell_type_names:
            if cell_type not in self.discovered_types:
                self._register_cell_type(cell_type)
        cell_types = torch.tensor(
            [self.cell_type_mapping[cell_type] for cell_type in cell_type_names],
            dtype=torch.long,
        )

        if self.subset_size is not None:
            number_cells = features.shape[0]
            if number_cells == 0:
                raise ValueError(f"Sample '{sample_name}' contains no cells")
            if number_cells > self.subset_size:
                indices = torch.randperm(number_cells)[: self.subset_size]
            elif number_cells < self.subset_size:
                indices = torch.randint(
                    0,
                    number_cells,
                    (self.subset_size,),
                )
            else:
                indices = torch.arange(number_cells)
            features = features[indices]
            cell_types = cell_types[indices]

        return features, cell_types, sample_name

    def get_cell_type_name(
        self,
        index: int,
    ) -> str:
        """Return the name corresponding to a cell-type index."""
        return self.reverse_mapping.get(index, f"Unknown_{index}")

    def get_cell_type_names(
        self,
        indices: Iterable[int] | torch.Tensor,
    ) -> list[str]:
        """Return names corresponding to cell-type indices."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        return [self.get_cell_type_name(index) for index in indices]

    def get_number_cell_types(self) -> int:
        """Return the number of registered cell types."""
        return len(self.cell_type_mapping)

    def get_num_cell_types(self) -> int:
        """Return the number of registered cell types."""
        return self.get_number_cell_types()

    def get_all_cell_types(self) -> list[str]:
        """Return all registered cell-type names."""
        return [
            self.reverse_mapping[index] for index in range(self.get_num_cell_types())
        ]

    def save_cell_type_mapping(
        self,
        filepath: str | Path,
    ) -> None:
        """Save the cell-type mapping as JSON."""
        with Path(filepath).open("w") as file:
            json.dump(
                {
                    "discovered_types": sorted(self.discovered_types),
                    "cell_type_mapping": self.cell_type_mapping,
                    "reverse_mapping": {
                        str(index): cell_type
                        for index, cell_type in self.reverse_mapping.items()
                    },
                },
                file,
                indent=2,
            )

    def load_cell_type_mapping(
        self,
        filepath: str | Path,
    ) -> None:
        """Load the cell-type mapping from JSON."""
        with Path(filepath).open() as file:
            data = json.load(file)
        self.discovered_types = set(data["discovered_types"])
        self.cell_type_mapping = data["cell_type_mapping"]
        self.reverse_mapping = {
            int(index): cell_type
            for index, cell_type in data["reverse_mapping"].items()
        }
