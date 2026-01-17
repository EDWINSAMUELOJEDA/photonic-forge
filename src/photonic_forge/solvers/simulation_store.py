"""Simulation Store - Persistent storage for simulation results.

Provides storage and query interface for simulation results,
enabling efficient data collection for training neural surrogates.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from photonic_forge.solvers.base import SimulationResult


class SimulationStore:
    """Persistent storage for simulation results.

    Uses SQLite for metadata and file system for permittivity arrays.
    Designed for efficient data collection and ML training.
    """

    def __init__(self, db_path: str | Path):
        """Initialize simulation store.

        Args:
            db_path: Path to SQLite database.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Permittivity arrays stored in same directory
        self.array_dir = self.db_path.parent / "permittivity_arrays"
        self.array_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    design_id TEXT UNIQUE NOT NULL,
                    geometry_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    wavelengths_json TEXT,
                    s_parameters_json TEXT,
                    predicted_metrics_json TEXT,
                    metadata_json TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_geometry_hash
                ON simulations(geometry_hash)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON simulations(created_at)
            """)

    def save(
        self,
        result: SimulationResult,
        permittivity: np.ndarray | None = None,
    ) -> int:
        """Save a simulation result.

        Args:
            result: SimulationResult to store.
            permittivity: Optional permittivity array to save.

        Returns:
            Database row ID.
        """
        # Convert S-parameters to JSON-serializable format
        s_params_serialized = {}
        for key, values in result.s_parameters.items():
            key_str = f"{key[0]}_{key[1]}"
            s_params_serialized[key_str] = {
                "real": np.real(values).tolist(),
                "imag": np.imag(values).tolist(),
            }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO simulations
                (design_id, geometry_hash, wavelengths_json, s_parameters_json,
                 predicted_metrics_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.design_id,
                    result.geometry_hash,
                    json.dumps(result.wavelengths.tolist()),
                    json.dumps(s_params_serialized),
                    json.dumps(result.predicted_metrics),
                    json.dumps(result.metadata),
                ),
            )
            row_id = cursor.lastrowid

        # Save permittivity array
        if permittivity is not None:
            array_path = self.array_dir / f"{result.design_id}.npy"
            np.save(array_path, permittivity)

        return row_id

    def load(self, design_id: str) -> SimulationResult | None:
        """Load a simulation result by design ID.

        Args:
            design_id: Unique design identifier.

        Returns:
            SimulationResult or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM simulations WHERE design_id = ?",
                (design_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_result(row)

    def _row_to_result(self, row: sqlite3.Row) -> SimulationResult:
        """Convert database row to SimulationResult."""
        wavelengths = np.array(json.loads(row["wavelengths_json"]))

        # Reconstruct S-parameters
        s_params_json = json.loads(row["s_parameters_json"])
        s_parameters = {}
        for key_str, values in s_params_json.items():
            parts = key_str.split("_")
            key = (parts[0], "_".join(parts[1:]))
            complex_values = np.array(values["real"]) + 1j * np.array(values["imag"])
            s_parameters[key] = complex_values

        return SimulationResult(
            s_parameters=s_parameters,
            wavelengths=wavelengths,
            design_id=row["design_id"],
            geometry_hash=row["geometry_hash"],
            predicted_metrics=json.loads(row["predicted_metrics_json"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def load_permittivity(self, design_id: str) -> np.ndarray | None:
        """Load permittivity array for a design.

        Args:
            design_id: Unique design identifier.

        Returns:
            Permittivity array or None if not found.
        """
        array_path = self.array_dir / f"{design_id}.npy"
        if array_path.exists():
            return np.load(array_path)
        return None

    def query(
        self,
        geometry_hash: str | None = None,
        min_date: datetime | None = None,
        max_date: datetime | None = None,
        limit: int = 100,
    ) -> list[SimulationResult]:
        """Query simulation results.

        Args:
            geometry_hash: Filter by geometry hash.
            min_date: Minimum creation date.
            max_date: Maximum creation date.
            limit: Maximum results to return.

        Returns:
            List of matching SimulationResults.
        """
        query = "SELECT * FROM simulations WHERE 1=1"
        params = []

        if geometry_hash:
            query += " AND geometry_hash = ?"
            params.append(geometry_hash)

        if min_date:
            query += " AND created_at >= ?"
            params.append(min_date.isoformat())

        if max_date:
            query += " AND created_at <= ?"
            params.append(max_date.isoformat())

        query += f" ORDER BY created_at DESC LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_result(row) for row in rows]

    def count(self) -> int:
        """Get total number of stored simulations."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
        return count

    def export_for_training(
        self,
        output_dir: str | Path,
        metric_names: list[str] | None = None,
    ) -> int:
        """Export data in format suitable for neural model training.

        Creates (permittivity.npy, metrics.json) pairs for each simulation.

        Args:
            output_dir: Output directory.
            metric_names: Specific metrics to export.

        Returns:
            Number of samples exported.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metric_names = metric_names or [
            "transmission",
            "insertion_loss_db",
            "bandwidth_3db",
            "return_loss_db",
        ]

        exported = 0
        for result in self.query(limit=10000):
            permittivity = self.load_permittivity(result.design_id)
            if permittivity is None:
                continue

            # Save permittivity
            np.save(output_path / f"sim_{exported:04d}.npy", permittivity)

            # Save log with metrics
            log_data = {
                "design_id": result.design_id,
                "geometry_hash": result.geometry_hash,
                "predicted_metrics": {
                    name: result.predicted_metrics.get(name, 0.0)
                    for name in metric_names
                },
            }
            with open(output_path / f"sim_{exported:04d}.json", "w") as f:
                json.dump(log_data, f, indent=2)

            exported += 1

        return exported

    def iter_all(self) -> Iterator[SimulationResult]:
        """Iterate through all stored simulations."""
        offset = 0
        batch_size = 100

        while True:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"SELECT * FROM simulations LIMIT {batch_size} OFFSET {offset}"
                ).fetchall()

            if not rows:
                break

            for row in rows:
                yield self._row_to_result(row)

            offset += batch_size


__all__ = ["SimulationStore"]
