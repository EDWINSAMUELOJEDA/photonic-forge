"""PhotonicForge Streamlit Web Application.

A visual interface for designing and optimizing photonic devices.
Launch with: streamlit run src/photonic_forge/ui/app.py
"""

import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="PhotonicForge",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Try imports - handle gracefully if not fully installed
try:
    from photonic_forge.core import (
        SILICON,
        SILICON_DIOXIDE,
        Circle,
        Rectangle,
        Waveguide,
        union,
    )
    from photonic_forge.core.geometry import Bend90, DirectionalCoupler
    HAS_CORE = True
except ImportError:
    HAS_CORE = False


def main():
    """Main application entry point."""
    st.title("🔬 PhotonicForge")
    st.markdown("**Photonic Integrated Circuit Design Platform**")

    if not HAS_CORE:
        st.error("PhotonicForge core module not found. Please install the package.")
        st.code("pip install -e .")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Geometry Builder", "Optimization", "Results Viewer", "About"],
    )

    if page == "Geometry Builder":
        geometry_builder_page()
    elif page == "Optimization":
        optimization_page()
    elif page == "Results Viewer":
        results_page()
    else:
        about_page()


def geometry_builder_page():
    """Interactive geometry builder."""
    st.header("🏗️ Geometry Builder")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Add Components")

        component_type = st.selectbox(
            "Component Type",
            ["Waveguide", "Rectangle", "Circle", "Bend90", "Directional Coupler"],
        )

        # Component-specific parameters
        if component_type == "Waveguide":
            st.markdown("**Waveguide Parameters**")
            x_start = st.number_input("Start X (µm)", value=0.0, step=1.0)
            y_start = st.number_input("Start Y (µm)", value=0.0, step=0.5)
            x_end = st.number_input("End X (µm)", value=20.0, step=1.0)
            y_end = st.number_input("End Y (µm)", value=0.0, step=0.5)
            width = st.number_input("Width (nm)", value=500, step=10)

            if st.button("Add Waveguide"):
                wg = Waveguide(
                    start=(x_start * 1e-6, y_start * 1e-6),
                    end=(x_end * 1e-6, y_end * 1e-6),
                    width=width * 1e-9,
                )
                if "components" not in st.session_state:
                    st.session_state.components = []
                st.session_state.components.append(("Waveguide", wg))
                st.success("Added waveguide!")

        elif component_type == "Rectangle":
            st.markdown("**Rectangle Parameters**")
            cx = st.number_input("Center X (µm)", value=0.0, step=1.0)
            cy = st.number_input("Center Y (µm)", value=0.0, step=0.5)
            w = st.number_input("Width (µm)", value=5.0, step=0.5)
            h = st.number_input("Height (µm)", value=2.0, step=0.5)

            if st.button("Add Rectangle"):
                rect = Rectangle(
                    center=(cx * 1e-6, cy * 1e-6),
                    width=w * 1e-6,
                    height=h * 1e-6,
                )
                if "components" not in st.session_state:
                    st.session_state.components = []
                st.session_state.components.append(("Rectangle", rect))
                st.success("Added rectangle!")

        elif component_type == "Circle":
            st.markdown("**Circle Parameters**")
            cx = st.number_input("Center X (µm)", value=0.0, step=1.0)
            cy = st.number_input("Center Y (µm)", value=0.0, step=0.5)
            r = st.number_input("Radius (µm)", value=5.0, step=0.5)

            if st.button("Add Circle"):
                circ = Circle(
                    center=(cx * 1e-6, cy * 1e-6),
                    radius=r * 1e-6,
                )
                if "components" not in st.session_state:
                    st.session_state.components = []
                st.session_state.components.append(("Circle", circ))
                st.success("Added circle!")

        elif component_type == "Directional Coupler":
            st.markdown("**Coupler Parameters**")
            length = st.number_input("Length (µm)", value=10.0, step=1.0)
            gap = st.number_input("Gap (nm)", value=200, step=10)
            width = st.number_input("Width (nm)", value=500, step=10)

            if st.button("Add Coupler"):
                coupler = DirectionalCoupler(
                    length=length * 1e-6,
                    gap=gap * 1e-9,
                    width=width * 1e-9,
                    center=(0, 0),
                )
                if "components" not in st.session_state:
                    st.session_state.components = []
                st.session_state.components.append(("Coupler", coupler))
                st.success("Added coupler!")

        # Clear button
        if st.button("Clear All"):
            st.session_state.components = []
            st.rerun()

    with col2:
        st.subheader("Preview")

        # Get components
        components = st.session_state.get("components", [])

        if components:
            # Combine all components
            shapes = [comp[1] for comp in components]
            if len(shapes) == 1:
                combined = shapes[0]
            else:
                combined = union(*shapes)

            # Generate visualization
            resolution = st.slider("Resolution (nm)", 20, 100, 50)

            # Auto-compute bounds
            x_min = st.number_input("X min (µm)", value=-5.0)
            x_max = st.number_input("X max (µm)", value=25.0)
            y_min = st.number_input("Y min (µm)", value=-5.0)
            y_max = st.number_input("Y max (µm)", value=5.0)

            bounds = (x_min * 1e-6, y_min * 1e-6, x_max * 1e-6, y_max * 1e-6)

            try:
                eps = combined.to_permittivity(
                    bounds=bounds,
                    resolution=resolution * 1e-9,
                    material_inside=SILICON,
                    material_outside=SILICON_DIOXIDE,
                )

                # Plot
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.imshow(
                    eps,
                    extent=[x_min, x_max, y_min, y_max],
                    origin="lower",
                    cmap="viridis",
                    aspect="auto",
                )
                ax.set_xlabel("X (µm)")
                ax.set_ylabel("Y (µm)")
                ax.set_title("Permittivity Map")
                plt.colorbar(im, ax=ax, label="ε_r")

                st.pyplot(fig)
                plt.close()

                st.info(f"Grid size: {eps.shape}")

            except Exception as e:
                st.error(f"Error generating preview: {e}")

            # Show component list
            st.markdown("**Components:**")
            for i, (name, _) in enumerate(components):
                st.write(f"{i+1}. {name}")
        else:
            st.info("Add components to see preview")


def optimization_page():
    """Optimization workflow page."""
    st.header("⚙️ Optimization")

    st.markdown("""
    Configure and run optimization for your photonic device.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Objective")

        objective_type = st.selectbox(
            "Optimization Goal",
            [
                "Minimize Insertion Loss",
                "Maximize Transmission",
                "Maximize Bandwidth",
                "Target Coupling Ratio",
            ],
        )

        if objective_type == "Target Coupling Ratio":
            target = st.slider("Target Coupling (%)", 0, 100, 50) / 100
            st.write(f"Target: {target:.0%}")

        st.subheader("Algorithm")

        algorithm = st.selectbox(
            "Optimizer",
            ["L-BFGS-B", "Nelder-Mead", "Genetic Algorithm", "Particle Swarm"],
        )

        max_iter = st.slider("Max Iterations", 10, 500, 100)

    with col2:
        st.subheader("Constraints")

        use_min_feature = st.checkbox("Minimum Feature Size", value=True)
        if use_min_feature:
            min_feature = st.number_input("Min Feature (nm)", value=100, step=10)

        use_symmetry = st.checkbox("Enforce Symmetry", value=False)
        if use_symmetry:
            sym_type = st.selectbox("Symmetry Type", ["Y-axis", "X-axis", "Both"])

        st.subheader("Parameters")

        param_type = st.selectbox(
            "Parameterization",
            ["Shape (gap, length, width)", "Pixel-based"],
        )

    st.markdown("---")

    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Optimizing..."):
            # Placeholder for actual optimization
            progress = st.progress(0)
            for i in range(100):
                import time
                time.sleep(0.02)
                progress.progress(i + 1)

            st.success("Optimization complete!")
            st.balloons()

            # Show mock results
            st.subheader("Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Objective", "0.0324", "-0.12")
            col2.metric("Coupling", "49.8%", "+0.2%")
            col3.metric("Iterations", "87")


def results_page():
    """Results viewer page."""
    st.header("📊 Results Viewer")

    st.info("Upload simulation results or select from history.")

    # Placeholder for results visualization
    st.subheader("S-Parameters")

    # Generate sample data
    wavelengths = np.linspace(1.5, 1.6, 100)
    s21 = 0.9 * np.exp(-((wavelengths - 1.55) ** 2) / 0.01**2)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(wavelengths * 1e3, 20 * np.log10(s21 + 1e-10))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("|S21| (dB)")
    ax.set_title("Transmission")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()


def about_page():
    """About page."""
    st.header("ℹ️ About PhotonicForge")

    st.markdown("""
    **PhotonicForge** is an open-source platform for photonic integrated circuit design.

    ### Features
    - 🏗️ SDF-based geometry primitives
    - ⚙️ Multiple optimization algorithms
    - 📊 Photonic metrics calculation
    - 📁 GDS export for fabrication

    ### Links
    - [GitHub Repository](https://github.com/edwinsamuelojeda/photonic-forge)
    - [Documentation](https://photonic-forge.readthedocs.io)

    ### License
    Apache 2.0
    """)

    st.markdown("---")
    st.caption("Built with Streamlit • PhotonicForge v0.1.0")


if __name__ == "__main__":
    main()
