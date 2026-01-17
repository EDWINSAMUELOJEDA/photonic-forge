"""WebGPU Exporter for PhotonicForge.

exports simulations to standalone HTML files with embedded WebGPU solvers.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


class WebGPUExporter:
    """Exports simulation to WebGPU HTML bundle."""

    def __init__(self, template_path: str | None = None, shader_path: str | None = None):
        current_dir = Path(__file__).parent
        self.template_path = Path(template_path or (current_dir / "template.html"))
        self.shader_path = Path(shader_path or (current_dir / "fdtd.wgsl"))

    def export(
        self,
        epsilon_grid: np.ndarray,
        output_path: str | Path,
        ds: float = 1.0, # dummy spatial step
        dt: float = 0.5, # dummy time step
    ) -> Path:
        """Export simulation data to HTML.
        
        Args:
            epsilon_grid: 2D numpy array of permittivity.
            output_path: Where to save the HTML.
            ds: Grid spacing.
            dt: Time step.
        """
        nx, ny = epsilon_grid.shape
        import pkgutil
        
        # Read files
        template = self.template_path.read_text("utf-8")
        shader = self.shader_path.read_text("utf-8")

        # Prepare Data
        # Flatten and convert to list for JSON serialization
        # TODO: For large grids, use base64 encoded binary blob
        eps_flat = epsilon_grid.astype(np.float32).flatten().tolist()
        
        data = {
            "params": {
                "nx": int(nx),
                "ny": int(ny),
                "dx": float(ds),
                "dt": float(dt)
            },
            "epsilon": eps_flat
        }
        
        # Inject
        html = template.replace("/* INJECT_SHADER_HERE */", shader)
        html = html.replace("/* INJECT_DATA_HERE */", json.dumps(data))
        
        out_path = Path(output_path)
        out_path.write_text(html, "utf-8")
        
        return out_path
