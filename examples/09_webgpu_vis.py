"""Example of WebGPU export.

Generates a simple waveguide geometry and exports it to a standalone HTML file
that runs a real-time FDTD simulation in the browser.
"""

import numpy as np
from photonic_forge.core import Waveguide, SILICON, SILICON_DIOXIDE
from photonic_forge.vis import WebGPUExporter
from pathlib import Path

def run_export():
    print("Generating geometry...")
    # 1. Create Layout
    wg = Waveguide(start=(0, 0), end=(10e-6, 0), width=0.5e-6)
    
    # 2. Rasterize to grid
    # Low res for fast web test: 200x100 grid
    bounds = (-2e-6, -2e-6, 12e-6, 2e-6)
    resolution = 50e-9 # 50nm
    
    eps_grid = wg.to_permittivity(
        bounds=bounds,
        resolution=resolution,
        material_inside=SILICON,
        material_outside=SILICON_DIOXIDE
    )
    
    print(f"Grid shape: {eps_grid.shape}")
    
    # 3. Export
    exporter = WebGPUExporter()
    out_file = Path("09_webgpu_vis.html")
    
    # Compute stability (Courant)
    # dt < dx / (c * sqrt(2)) -> normalized units where c=1
    dx = resolution
    dt = dx * 0.5 
    
    print(f"Exporting to {out_file}...")
    exporter.export(
        epsilon_grid=eps_grid,
        output_path=out_file,
        ds=dx,
        dt=dt
    )
    print("Done! Open 09_webgpu_vis.html in a WebGPU-enabled browser (Chrome/Edge).")

if __name__ == "__main__":
    run_export()
