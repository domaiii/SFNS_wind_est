import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.tri as mtri

from scipy.spatial import cKDTree
from pathlib import Path
from basix.ufl import element
from dolfinx import fem, mesh

class Visualizer:
    """
    2D visualizer for FEM meshes and functions using matplotlib.
    """

    def __init__(self, mesh: mesh.Mesh, figsize=(10, 5), dpi=160):
        self.mesh = mesh
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self._last_mappable = None

        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, 0)
        conn = self.mesh.topology.connectivity(tdim, 0).array

        num_cells = len(conn) // 3
        self.cells = conn.reshape(num_cells, 3)
        self.points = self.mesh.geometry.x[:, :2]

    def add_background_mesh(self, color="0.75", linewidth=0.25, alpha=0.9):
        x = self.points[:, 0]
        y = self.points[:, 1]
        self.ax.triplot(x, y, self.cells, color=color, linewidth=linewidth, alpha=alpha)

    def add_scalar_field(self, name: str, scalar_func: fem.Function, cmap: str = "coolwarm"):
        bs = scalar_func.function_space.dofmap.index_map_bs
        if not bs == 1:
            raise ValueError(f"{name} must be a scalar field (block size = 1).")

        V_plot = fem.functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), 1))
        scalar_plot = fem.Function(V_plot)
        scalar_plot.interpolate(scalar_func)

        tri = mtri.Triangulation(self.points[:, 0], self.points[:, 1], self.cells)
        self._last_mappable = self.ax.tripcolor(
            tri,
            scalar_plot.x.array,
            shading="gouraud",
            cmap=cmap,
        )

    def add_vector_field(
        self,
        name: str,
        vector_func: fem.Function,
        stride: int = 2,
        cmap: str = "coolwarm",
        scale: float | None = None,
        width: float = 0.0022,
    ):
        bs = vector_func.function_space.dofmap.index_map_bs
        if bs < 2:
            raise ValueError(f"{name} must be a vector field with at least 2 components.")

        coords = vector_func.function_space.tabulate_dof_coordinates()[:, :2]
        values = vector_func.x.array.reshape(-1, bs)[:, :2]
        mag = np.linalg.norm(values, axis=1)

        stride = max(int(stride), 1)
        coords = coords[::stride]
        values = values[::stride]
        mag = mag[::stride]

        quiv = self.ax.quiver(
            coords[:, 0],
            coords[:, 1],
            values[:, 0],
            values[:, 1],
            mag,
            cmap=cmap,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=width,
            pivot="tail",
        )
        self._last_mappable = quiv

    def add_points(self, coords: np.ndarray, color="red", size=15, label: str | None = None):
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        self.ax.scatter(coords[:, 0], coords[:, 1], c=color, s=size, label=label)

    def draw_boundary_segment(
        self,
        segments: list[np.ndarray],
        color="black",
        linewidth: float = 2.0,
        label: str | None = None,
    ):
        """Add XY line segments to the plot."""
        plotted_label = label
        for segment in segments:
            xy = np.asarray(segment, dtype=float)
            if xy.ndim != 2 or xy.shape[1] < 2:
                raise ValueError("Each segment must have shape (n_points, >=2).")
            self.ax.plot(
                xy[:, 0],
                xy[:, 1],
                clip_on=False,
                color=color,
                linewidth=linewidth,
                label=plotted_label,
            )
            plotted_label = None

    def add_streamplot(
        self,
        vector_func: fem.Function,
        nx: int = 220,
        ny: int = 140,
        density: float = 2.0,
        cmap: str = "coolwarm",
        linewidth: float = 1.3,
        arrowsize: float = 1.0,
    ):
        """
        Plot streamlines for a 2D vector field on a regular grid.
        Grid points outside the mesh domain (including holes) are masked.
        """
        bs = vector_func.function_space.dofmap.index_map_bs
        if bs < 2:
            raise ValueError(f"Given field must be a vector field with at least 2 components.")

        # Regular plotting grid in domain bounding box
        x_min, y_min = np.min(self.points, axis=0)
        x_max, y_max = np.max(self.points, axis=0)
        xg = np.linspace(x_min, x_max, max(int(nx), 10))
        yg = np.linspace(y_min, y_max, max(int(ny), 10))
        xx, yy = np.meshgrid(xg, yg)
        q = np.column_stack([xx.ravel(), yy.ravel()])

        # Mark points outside mesh/hole regions
        tri = mtri.Triangulation(self.points[:, 0], self.points[:, 1], self.cells)
        tri_finder = tri.get_trifinder()
        inside = tri_finder(q[:, 0], q[:, 1]) >= 0

        # Interpolate from function DOF coordinates by nearest neighbor
        dof_xy = vector_func.function_space.tabulate_dof_coordinates()[:, :2]
        dof_uv = vector_func.x.array.reshape(-1, bs)[:, :2]
        tree = cKDTree(dof_xy)
        _, nn = tree.query(q[inside], k=1, workers=-1)

        U = np.full(q.shape[0], np.nan, dtype=float)
        V = np.full(q.shape[0], np.nan, dtype=float)
        U[inside] = dof_uv[nn, 0]
        V[inside] = dof_uv[nn, 1]

        U = U.reshape(xx.shape)
        V = V.reshape(xx.shape)
        speed = np.sqrt(U**2 + V**2)
        outside_mask = ~inside.reshape(xx.shape)

        strm = self.ax.streamplot(
            xg,
            yg,
            np.ma.array(U, mask=outside_mask),
            np.ma.array(V, mask=outside_mask),
            color=np.ma.array(speed, mask=outside_mask),
            cmap=cmap,
            density=density,
            linewidth=linewidth,
            arrowsize=arrowsize,
        )
        self._last_mappable = strm.lines

    def _prepare_plot(
        self,
        title: str | None = None,
        show_colorbar: bool = True,
        colorbar_label: str | None = "Wind speed magnitude (m/s)",
    ):
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        for spine in self.ax.spines.values():
            spine.set_color("0.75")
            spine.set_linewidth(0.7)
            spine.set_zorder(0)
        self.ax.tick_params(colors="0.35", width=0.7)
        if title is not None:
            self.ax.set_title(title)

        if show_colorbar and self._last_mappable is not None:
            cbar = self.fig.colorbar(self._last_mappable, ax=self.ax, pad=0.02)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label)

        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=max(1, len(labels)),
                frameon=False,
                borderaxespad=0.0,
            )

        self.fig.tight_layout()

    def show(
        self,
        title: str | None = None,
        show_colorbar: bool = True,
        colorbar_label: str | None = "Wind speed magnitude (m/s)",
    ):
        self._prepare_plot(
            title=title,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
        )
        plt.show()

    def save(
        self,
        filename: str | Path,
        title: str | None = None,
        show_colorbar: bool = True,
        colorbar_label: str | None = "Wind speed magnitude (m/s)",
    ):
        self._prepare_plot(
            title=title,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
        )
        self.fig.savefig(filename, dpi=self.fig.dpi)
        print(f"[Visualizer] Saved figure: {filename}")



def plot_wind_csv(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    stride: int = 1,
    z_height: float | None = None,
    z_tol: float = 0.05,
    z_span_threshold: float = 0.2,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 160,
    cmap: str = "coolwarm",
    scale: float | None = None,
    width: float = 0.0022,
    colorbar_label: str = "wind speed (m/s)",
    show: bool = True,
):
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path)
    required = ["Points:0", "Points:1", "U:0", "U:1"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Unsupported wind CSV format in {csv_path.name}. "
            "Expected columns Points:0,Points:1,U:0,U:1"
            f"; missing {', '.join(missing)}."
        )

    if "Points:2" in df.columns:
        z = df["Points:2"].to_numpy(dtype=float)
        if z_height is not None:
            mask = np.abs(z - float(z_height)) <= float(z_tol)
            df = df.loc[mask].copy()
            if df.empty:
                raise ValueError(
                    f"No CSV rows found in slice z={z_height:.6g} +/- {z_tol:.6g} "
                    f"for {csv_path.name}."
                )
        elif z.size > 0 and float(np.max(z) - np.min(z)) > float(z_span_threshold):
            raise ValueError(
                f"{csv_path.name} contains multiple z-levels. "
                "Pass z_height to choose the slice to plot."
            )

    stride = max(int(stride), 1)
    df = df.iloc[::stride].copy()

    x = df["Points:0"].to_numpy(dtype=float)
    y = df["Points:1"].to_numpy(dtype=float)
    u = df["U:0"].to_numpy(dtype=float)
    v = df["U:1"].to_numpy(dtype=float)
    speed = np.sqrt(u * u + v * v)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    quiv = ax.quiver(
        x,
        y,
        u,
        v,
        speed,
        cmap=cmap,
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=width,
        pivot="tail",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(quiv, ax=ax, pad=0.02)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=dpi)
        print(f"[plot_wind_csv] Saved figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize wind CSV files and save them as PNG plots.")
    parser.add_argument("windfile_csv", type=str, help="Path to the wind CSV file.")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Optional output directory for the PNG. Defaults to the input file's parent directory.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--stride", type=int, default=1, help="Plot every n-th row to reduce clutter.")
    parser.add_argument("--scale", type=float, default=None, help="Optional matplotlib quiver scale.")
    parser.add_argument("--width", type=float, default=0.0022, help="Arrow width for quiver plots.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(10, 5), help="Figure size in inches.")
    parser.add_argument("--cmap", type=str, default="coolwarm", help="Matplotlib colormap.")
    parser.add_argument("--show", action="store_true", help="Also show the figure interactively.")
    parser.add_argument("--z-height", type=float, default=None, help="Slice center height for CSV files with multiple z levels.")
    parser.add_argument("--z-tol", type=float, default=0.05, help="Half-thickness of the z slice, e.g. 0.05 means +/- 5 cm.")
    parser.add_argument("--z-span-threshold", type=float, default=0.2, help="Require --z-height if the z-span exceeds this threshold.")
    args = parser.parse_args()

    csv_path = Path(args.windfile_csv).resolve()
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Can only visualize .csv files.")

    output_dir = csv_path.parent if args.output_dir is None else Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_z{args.z_height:g}" if args.z_height is not None else ""
    output_path = output_dir / f"{csv_path.stem}{suffix}.png"

    plot_wind_csv(
        csv_path,
        output_path=output_path,
        title=args.title,
        stride=args.stride,
        z_height=args.z_height,
        z_tol=args.z_tol,
        z_span_threshold=args.z_span_threshold,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        cmap=args.cmap,
        scale=args.scale,
        width=args.width,
        show=args.show,
    )

