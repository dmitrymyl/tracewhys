from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist, squareform


def _coord_cols_for_dim(dim: Literal['2d', '3d']) -> list[str]:
    """Resolve the coordinate columns to use for a given dimensionality.

    Args:
        dim: Coordinate dimensionality, either '2d' or '3d'.

    Returns:
        list[str]: `['x', 'y']` for '2d', `['x', 'y', 'z']` for '3d'.

    Raises:
        ValueError: If `dim` is not '2d' or '3d'.
    """
    if dim == '2d':
        return ['x', 'y']
    elif dim == '3d':
        return ['x', 'y', 'z']
    raise ValueError(f'`dim` must be "2d" or "3d", but "{dim}" was provided.')


class Trace:
    """Handles coordinate measurements for a single continuous trace run.

    A `Trace` represents one ordered, independently-imputed run of
    timepoints -- the atomic building block other trace types are made of.
    It can be used directly through inheritance (as `RnaTrace` does, adding
    RNA-specific shape metrics) or as a component of a composite trace made
    of several independently-imputed runs (as `SisterTrace` does, via
    composition).

    Args:
        df (pd.DataFrame): Input coordinate measurements with a `timepoint`
            column and coordinate columns (`x`, `y`, [`z`] for 3D). May
            contain missing values.
        valid_timepoints (Sequence): Ordered list/array of expected timepoints.
        dim (str): Coordinate dimensionality, either '2d' or '3d'.
        impute (bool): If True, linearly impute interior missing coordinates.

    Attributes:
        original_timepoint_mask (np.ndarray): Boolean mask per timepoint indicating
            whether a coordinate was present in the original pre-imputation input.
        df (pd.DataFrame): Working dataframe (imputed if `impute=True`).
        dim (str): Coordinate dimensionality ('2d' or '3d').

    Notes:
        - Properties prefixed with `original_` refer to the pre-imputation state.
        - Metric properties return `numpy.nan` when required coordinates are missing.
        - Private helpers and cached attributes are named with a leading underscore and
          are implementation details (not part of the public API).
    """
    def __init__(self,
                 df: pd.DataFrame,
                 valid_timepoints: Sequence[int],
                 dim: Literal['2d', '3d'] = '2d',
                 impute: bool = True,
                ) -> None:
        """Create a trace run from coordinate measurements.

        Args:
            df: DataFrame containing timepoint-indexed coordinate measurements.
            valid_timepoints: Expected ordered timepoints for the trace.
            dim: Coordinate dimensionality, either ``"2d"`` or ``"3d"``.
            impute: Whether to linearly impute missing interior coordinates.

        Raises:
            ValueError: If ``dim`` is invalid or required columns/timepoints are
                missing or unexpected.
        """
        self._valid_timepoints: Sequence[int] = valid_timepoints
        self._coord_cols: list[str] = _coord_cols_for_dim(dim)
        self.dim: Literal['2d', '3d'] = dim

        self._validate_structure(df)
        self._original_df: pd.DataFrame = self._reindex_to_valid_timepoints(df)

        self._timepoint_mask: NDArray[np.bool_] | None = None

        if impute:
            self.df: pd.DataFrame = self._impute()
        else:
            self.df = self._original_df

        self._gyration_radius: float | None = None
        self._distance_matrix: NDArray[np.float64] | None = None

    def _validate_structure(self, df: pd.DataFrame) -> None:
        """Validate the input dataframe structure.

        Args:
            df: Input dataframe containing coordinate columns and a timepoint
                column.

        Raises:
            ValueError: If required columns are missing or unexpected
                timepoints are present.
        """
        failure_reasons = list()

        necessary_columns = self._coord_cols + ['timepoint']
        for col in necessary_columns:
            if col not in df:
                failure_reasons.append(f'column "{col}" is missing')

        actual_tps = set(df['timepoint'].values)
        exp_tps = set(self._valid_timepoints)
        unexp_tps = actual_tps - exp_tps
        if unexp_tps:
            failure_reasons.append(f'unexpected timepoints are detected: {unexp_tps}')

        if failure_reasons:
            raise ValueError(failure_reasons)

    def _reindex_to_valid_timepoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex a dataframe to the full set of valid timepoints.

        Args:
            df: Input dataframe indexable by ``timepoint``.

        Returns:
            A dataframe indexed by ``timepoint`` and reindexed to ``self._valid_timepoints``.
        """
        return df.set_index('timepoint').reindex(self._valid_timepoints)

    @property
    def original_timepoint_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of the original timepoint presence before imputation.

        Returns:
            np.ndarray: 1-D boolean array with length equal to the number of valid
                timepoints. True indicates the coordinate was present in the
                original pre-imputation input.
        """
        if self._timepoint_mask is None:
            self._timepoint_mask = (~self._original_df[self._coord_cols[0]].isna()).to_numpy()
        return self._timepoint_mask

    def _impute(self) -> pd.DataFrame:
        """Linearly impute interior missing coordinates in the original dataframe.

        Behavior:
            - If there are no missing timepoints in the unimputed input, returns
              the original reindexed dataframe unchanged.
            - Fills interior gaps by linear interpolation along each coordinate column
              using pandas' `interpolate` with `limit_area='inside'`.

        Returns:
            pd.DataFrame: DataFrame containing imputed coordinates.
        """
        new_df = self._original_df.copy()
        for coord in self._coord_cols:
            new_df[coord] = new_df[coord].interpolate(method='linear', limit_area='inside')
        return new_df.reset_index()

    @property
    def _coords(self) -> NDArray[np.float64]:
        """Return the working coordinates as a NumPy array.

        Returns:
            np.ndarray: Array of shape (n_timepoints, n_dimensions) containing
                coordinates from the current `df` (imputed or original depending
                on initialization).
        """
        return self.df[self._coord_cols].to_numpy()

    @property
    def first_coord(self) -> NDArray[np.float64]:
        """Coordinate of the first timepoint in this trace run."""
        return self._coords[0]

    @property
    def last_coord(self) -> NDArray[np.float64]:
        """Coordinate of the last timepoint in this trace run."""
        return self._coords[-1]

    @property
    def _has_remaining_missing_edges(self) -> bool:
        """Whether any edge (first or last) coordinate remains missing after imputation.

        Returns:
            bool: True if either the first or last coordinate row contains NaNs.
        """
        return bool(np.isnan(self._coords[[0, -1], :]).any())

    @property
    def _has_remaining_missing_segments(self) -> bool:
        """Whether any coordinate in the trace remains missing after imputation.

        Returns:
            bool: True if there are any NaNs in the coordinate array.
        """
        return bool(np.isnan(self._coords).any())

    @staticmethod
    def _has_missing_values(arr: ArrayLike) -> bool:
        """Check whether the array contains any missing (NaN) values.

        Args:
            arr: Array-like object.

        Returns:
            bool: True if `arr` contains at least one NaN, False otherwise.
        """
        return bool(np.isnan(arr).any())

    @classmethod
    def _compute_gyration_radius(cls, coords: NDArray[np.float64]) -> float:
        """Compute the radius of gyration for a set of coordinates.

        Args:
            coords (np.ndarray): Array of shape (n, d) where n is the number of
                points and d is dimensionality (2 or 3).

        Returns:
            float: Radius of gyration, or `numpy.nan` if `coords` contains NaNs.
        """
        if cls._has_missing_values(coords):
            return np.nan
        center_of_mass = np.mean(coords, axis=0)
        shifted_coords = coords - center_of_mass
        gyration_radius = np.sqrt(np.mean(np.sum(shifted_coords ** 2, axis=1)))
        return float(gyration_radius)

    @property
    def gyration_radius(self) -> float:
        """Radius of gyration of the trace coordinates.

        Returns:
            float: Computed radius of gyration, or `numpy.nan` if any timepoints
                are missing from the current `df`.
        """
        if self._gyration_radius is None:
            self._gyration_radius = self._compute_gyration_radius(self._coords)
        return float(self._gyration_radius)

    @property
    def distance_matrix(self) -> NDArray[np.float64]:
        """Pairwise Euclidean distance matrix between timepoints.

        Returns:
            np.ndarray: Symmetric (n x n) matrix of pairwise distances.
        """
        if self._distance_matrix is None:
            self._distance_matrix = squareform(pdist(self._coords, metric='euclidean'))
        return self._distance_matrix


class RnaTrace(Trace):
    """Handles coordinate measurements for a single RNA trace and computes shape metrics.

    Args:
        df (pd.DataFrame): Input coordinate measurements with a `timepoint` column
            and coordinate columns (`x`, `y`, [`z`] for 3D). May contain missing values.
        valid_timepoints (Sequence): Ordered list/array of expected timepoints.
        dim (str): Coordinate dimensionality, either '2d' or '3d'.
        impute (bool): If True, linearly impute interior missing coordinates.
        metadata (TraceMetadata | None): Optional metadata for the trace.

    Attributes:
        metadata (dict | None): Trace metadata if provided.

    Notes:
        - See `Trace` for the shared single-run behavior: imputation,
          `original_timepoint_mask`, `gyration_radius`, `distance_matrix`, etc.
        - Metric properties return `numpy.nan` when required coordinates are missing.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 valid_timepoints: Sequence[int],
                 dim: Literal['2d', '3d'] = '2d',
                 impute: bool = True,
                 metadata: dict | None = None,
                ) -> None:
        """Create an RNA trace from coordinate measurements.

        Args:
            df: DataFrame containing timepoint-indexed coordinate measurements.
            valid_timepoints: Expected ordered timepoints for the trace.
            dim: Coordinate dimensionality, either ``"2d"`` or ``"3d"``.
            impute: Whether to linearly impute missing interior coordinates.
            metadata: Optional trace metadata.

        Raises:
            ValueError: If ``dim`` is invalid or required columns/timepoints are
                missing or unexpected.
        """
        super().__init__(df, valid_timepoints, dim=dim, impute=impute)
        self.metadata: dict | None = metadata
        self._convex_hull: ConvexHull | None = None

    @property
    def original_present_count(self) -> int:
        """Number of timepoints present in the original input.

        Returns:
            int: Count of timepoints that had coordinate data before imputation.
        """
        return int(self.original_timepoint_mask.sum())

    @property
    def original_missing_count(self) -> int:
        """Number of timepoints missing in the original input.

        Returns:
            int: Count of timepoints that were missing before imputation.
        """
        return int((~self.original_timepoint_mask).sum())

    @property
    def has_original_missing_left_edge(self) -> bool:
        """Whether the first timepoint was missing in the original input.

        Returns:
            bool: True if the coordinate at the first timepoint was missing before
                imputation, False otherwise.
        """
        return not self.original_timepoint_mask[0]

    @property
    def has_original_missing_right_edge(self) -> bool:
        """Whether the last timepoint was missing in the original input.

        Returns:
            bool: True if the coordinate at the last timepoint was missing before
                imputation, False otherwise.
        """
        return not self.original_timepoint_mask[-1]

    @property
    def has_original_missing_left_flank(self) -> bool:
        """Whether the first two timepoints were both missing in the original input.

        Returns:
            bool: True if both the first and second timepoints were missing before
                imputation, False otherwise.
        """
        return not self.original_timepoint_mask[0] and not self.original_timepoint_mask[1]

    @property
    def has_original_missing_right_flank(self) -> bool:
        """Whether the last two timepoints were both missing in the original input.

        Returns:
            bool: True if both the penultimate and last timepoints were missing before
                imputation, False otherwise.
        """
        return not self.original_timepoint_mask[-2] and not self.original_timepoint_mask[-1]

    @property
    def has_original_consecutive_missing_timepoints(self) -> bool:
        """Whether the original input contains consecutive missing timepoints.

        Returns:
            bool: True if there exists at least one pair of adjacent timepoints both
                missing before imputation, False otherwise.
        """
        missing = ~self.original_timepoint_mask
        adjacent_missing_pairs = missing[:-1] & missing[1:]
        return bool(adjacent_missing_pairs.any())

    @property
    def half_extent_asymmetry(self) -> float:
        """Half-extent asymmetry comparing left and right halves of a trace.

        The metric compares the spatial extent of the left and right halves
        by computing the radius of gyration for each half and returning the
        normalized difference:

            (Rg_right - Rg_left) / (Rg_right + Rg_left)

        where Rg_left is computed over the first floor(N/2) points and
        Rg_right over the last ceil(N/2) points of the trace. The metric is
        signed: positive values indicate the right half is more extended,
        negative values indicate the left half is more extended. Values are
        typically in the range (-1, 1) when radii are positive and finite.

        Returns:
            float: The half-extent asymmetry, or ``numpy.nan`` if the metric
                cannot be computed because of missing coordinates. If both
                radii are zero the result is undefined.
        """
        if self._has_remaining_missing_segments:
            return np.nan
        coords = self._coords
        N = coords.shape[0]
        rg_left = self._compute_gyration_radius(coords[:N // 2, :])
        rg_right = self._compute_gyration_radius(coords[N - N // 2:, :])
        return float((rg_right - rg_left) / (rg_right + rg_left))

    @property
    def end_to_end_distance(self) -> float:
        """Euclidean distance between the first and last timepoints.

        Returns:
            float: Distance between endpoints, or `numpy.nan` if either edge
                coordinate is missing from the current `df`.
        """
        if self._has_remaining_missing_edges:
            return np.nan
        return self.distance_matrix[0, -1]

    @property
    def contour_length(self) -> float:
        """Total contour length computed as the sum of consecutive segment lengths.

        Returns:
            float: Sum of distances between successive timepoints, or `numpy.nan`
                if any timepoint is missing from the current `df`.
        """
        if self._has_remaining_missing_segments:
            return np.nan
        return float(np.sum(np.diag(self.distance_matrix, k=1)))

    @property
    def convex_hull(self) -> ConvexHull | None:
        """Convex hull of the coordinate set (2D case).

        Returns:
            scipy.spatial.ConvexHull or None: ConvexHull object when coordinates
                are complete; None if there are missing segments from the current `df`.
        """
        if self._has_remaining_missing_segments:
            return None
        if self._convex_hull is None:
            try:
                self._convex_hull = ConvexHull(self._coords)
            except QhullError:
                return None
        return self._convex_hull

    def perimeter_2d(self) -> float:
        """Perimeter of the convex hull in 2D.

        Returns:
            float: Perimeter length if `dim == '2d'` and a valid convex hull
            can be computed; otherwise ``numpy.nan``.
        """
        if self.dim != '2d' or self.convex_hull is None:
            return np.nan
        return float(self.convex_hull.area)

    def area_2d(self) -> float:
        """Area of the convex hull in 2D.

        Returns:
            float: Polygon area (convex hull volume attribute) if `dim == '2d'`;
            otherwise ``numpy.nan``.
        """
        if self.dim != '2d' or self.convex_hull is None:
            return np.nan
        return float(self.convex_hull.volume)

    def surface_area_3d(self) -> float:
        """Surface area of the convex hull in 3D.

        Returns:
            float: Surface area if `dim == '3d'` and a valid convex hull can be
            computed; otherwise ``numpy.nan``.
        """
        if self.dim != '3d' or self.convex_hull is None:
            return np.nan
        return float(self.convex_hull.area)

    def volume_3d(self) -> float:
        """Volume enclosed by the convex hull in 3D.

        Returns:
            float: Volume if `dim == '3d'` and a valid convex hull can be
            computed; otherwise ``numpy.nan``.
        """
        if self.dim != '3d' or self.convex_hull is None:
            return np.nan
        return float(self.convex_hull.volume)

    @property
    def perimeter(self) -> float:
        """Convenience property for perimeter.

        Returns the convex-hull perimeter when `dim == '2d'`, otherwise
        returns ``numpy.nan``.
        """
        if self.dim == '2d':
            return self.perimeter_2d()
        return np.nan

    @property
    def area(self) -> float:
        """Convenience property for area/surface area.

        - If `dim == '2d'` returns the 2D polygon area (convex hull area).
        - If `dim == '3d'` returns the surface area of the 3D convex hull.
        Returns ``numpy.nan`` when unavailable.
        """
        if self.dim == '2d':
            return self.area_2d()
        if self.dim == '3d':
            return self.surface_area_3d()
        return np.nan

    @property
    def volume(self) -> float:
        """Convenience property for volume.

        Returns the 3D convex-hull volume when `dim == '3d'`, otherwise
        returns ``numpy.nan``.
        """
        if self.dim == '3d':
            return self.volume_3d()
        return np.nan

    @property
    def asphericity(self) -> float:
        """Asphericity of the trace based on its gyration tensor.

        The metric is computed from the normalized inertia (gyration) tensor of the
        coordinate cloud and quantifies how far the shape deviates from a sphere.
        Larger values indicate a more anisotropic distribution of mass.

        Returns:
            float: Asphericity value, or ``numpy.nan`` if any coordinate in the
                current trace is missing.
        """
        if self._has_remaining_missing_segments:
            return np.nan
        c = self._coords
        d = c.shape[1]                 # Auto-detects dimension (2 or 3)

        c = c - c.mean(axis=0)         # 1. Shift to center of mass
        S = (c.T @ c) / len(c)         # 2. Compute gyration tensor

        tr_S = np.trace(S)             # Trace of S
        tr_S2 = np.trace(S @ S)        # Trace of S squared matrix

        # 3. Generalized asphericity formula
        return float((d * tr_S2 - tr_S**2) / ((d - 1) * tr_S**2))

    def get_scaling_table(self,
                          mode: Literal['anchor_5', 'anchor_3', 'all', 'agg'] = 'all',
                          agg_func: Callable[[NDArray[np.float64]], float] | None = None,
                         ) -> pd.DataFrame:
        """Build a distance table for scaling-curve calculations.

        Args:
            mode: One of the following:
                "anchor_5": Distances from the 5' end (index 0) to each
                    downstream timepoint.
                "anchor_3": Distances from the 3' end (last index) to each
                    upstream timepoint.
                "all": All pairwise distances grouped by genomic/timepoint
                    separation.
                "agg": One aggregated distance per separation, using
                    ``agg_func``.
            agg_func: Aggregation function used when ``mode`` is ``"agg"``.
                Defaults to ``np.nanmedian``.

        Returns:
            pd.DataFrame: DataFrame with columns ``separation`` and ``dist``.
        """
        mtx = self.distance_matrix
        n = mtx.shape[0]
        if mode == 'anchor_5':
            records = [{'separation': sep, "dist": mtx[0, sep]}
                       for sep in range(1, n)]
        elif mode == 'anchor_3':
            records = [{'separation': sep, "dist": mtx[-1, n - 1 - sep]}
                       for sep in range(1, n)]
        else:
            sep_to_dist = {sep: mtx.diagonal(sep)
                           for sep in range(1, n)}
            if mode == 'all':
                records = [{'separation': sep, "dist": dist}
                           for sep, dists in sep_to_dist.items()
                           for dist in dists]
            elif mode == 'agg':
                if agg_func is None:
                    agg_func = np.nanmedian
                records = [{'separation': sep, "dist": agg_func(dists)}
                           for sep, dists in sep_to_dist.items()]
            else:
                raise ValueError(f'`mode` must be one of "anchor_5", "anchor_3", "all", "agg", '
                                 f'but "{mode}" was provided.'
                                )
        return pd.DataFrame.from_records(records)

    def get_all_metrics(self):
        """Compute and return all available metrics for the trace."""
        scaling_df = self.get_scaling_table(mode='all')
        scaling_coef = stats.linregress(np.log10(scaling_df['separation']),
                                        np.log10(scaling_df['dist']))\
                            .slope

        metrics = dict(Rg=self.gyration_radius,
                       asphericity=self.asphericity,
                       end_to_end_dist=self.end_to_end_distance,
                       contour_length=self.contour_length,
                       end_to_end_dist_norm=self.end_to_end_distance / self.contour_length,
                       half_ext_asym=self.half_extent_asymmetry,
                       scaling_exp=scaling_coef,
                       perimeter=self.perimeter,
                       area=self.area,
                       volume=self.volume)

        dropouts_info = dict(missing_count=self.original_missing_count,
                             present_count=self.original_present_count,
                             consecutive_missing=self.has_original_consecutive_missing_timepoints,
                             missing_left_edge=self.has_original_missing_left_edge,
                             missing_right_edge=self.has_original_missing_right_edge,
                             missing_left_flank=self.has_original_missing_left_flank,
                             missing_right_flank=self.has_original_missing_right_flank,
                             timepoint_mask=self.original_timepoint_mask)

        properties = dict(dim=self.dim,
                          distance_matrix=self.distance_matrix)

        total_info = metrics | dropouts_info | self.metadata | properties
        return total_info


class SisterTrace:
    """Handles coordinate measurements for a pair of sister-chromatid traces.

    Composed of four `Trace` segments: sister1/sister2, each split into a
    left and right side relative to the DNA-repair cut site. Every segment
    has its own set of valid timepoints, and missing coordinates are imputed
    only within a segment -- never across the cut site or between sisters.

    Args:
        df (pd.DataFrame): Input coordinate measurements with a `timepoint`
            column and coordinate columns (`x`, `y`, [`z`] for 3D). May
            contain missing values.
        s1_left_timepoints (Sequence[int]): Valid timepoints for sister1, left
            of the cut site.
        s1_right_timepoints (Sequence[int]): Valid timepoints for sister1,
            right of the cut site.
        s2_left_timepoints (Sequence[int]): Valid timepoints for sister2, left
            of the cut site.
        s2_right_timepoints (Sequence[int]): Valid timepoints for sister2,
            right of the cut site.
        dim (str): Coordinate dimensionality, either '2d' or '3d'.
        impute (bool): If True, linearly impute interior missing coordinates
            within each segment.
        metadata (dict | None): Optional metadata for the trace.

    Attributes:
        df (pd.DataFrame): Combined dataframe (concatenation of the four
            segments' data, sorted by timepoint), with `sister` and `side`
            columns tagging each row.
        dim (str): Coordinate dimensionality ('2d' or '3d').
        metadata (dict | None): Trace metadata if provided.
    """
    _SISTERS = ('sister1', 'sister2')
    _SIDES = ('left', 'right')
    _CUT_EDGE_ATTR = {'left': 'last_coord', 'right': 'first_coord'}

    def __init__(self,
                 df: pd.DataFrame,
                 s1_left_timepoints: Sequence[int],
                 s1_right_timepoints: Sequence[int],
                 s2_left_timepoints: Sequence[int],
                 s2_right_timepoints: Sequence[int],
                 dim: Literal['2d', '3d'] = '2d',
                 impute: bool = True,
                 metadata: dict | None = None,
                ) -> None:
        self.metadata: dict | None = metadata
        self._coord_cols: list[str] = _coord_cols_for_dim(dim)
        self.dim: Literal['2d', '3d'] = dim

        valid_timepoints_by_key: dict[tuple[str, str], Sequence[int]] = {
            ('sister1', 'left'): s1_left_timepoints,
            ('sister1', 'right'): s1_right_timepoints,
            ('sister2', 'left'): s2_left_timepoints,
            ('sister2', 'right'): s2_right_timepoints,
        }
        self._validate_structure(df, valid_timepoints_by_key)

        self._segments: dict[tuple[str, str], Trace] = {
            key: Trace(df[df['timepoint'].isin(tps)], tps, dim=dim, impute=impute)
            for key, tps in valid_timepoints_by_key.items()
        }

        self.df: pd.DataFrame = self._build_combined_df()

        self._distance_matrix: NDArray[np.float64] | None = None
        self._cut_end_dist_mtx: NDArray[np.float64] | None = None

    def _validate_structure(self,
                            df: pd.DataFrame,
                            valid_timepoints_by_key: dict[tuple[str, str], Sequence[int]],
                           ) -> None:
        """Validate the input dataframe and the four timepoint segments.

        Raises:
            ValueError: If required columns are missing, a timepoint is
                assigned to more than one sister/side segment, or `df`
                contains timepoints outside all four segments.
        """
        failure_reasons = list()

        for col in self._coord_cols + ['timepoint']:
            if col not in df:
                failure_reasons.append(f'column "{col}" is missing')

        seen_tps = set()
        overlapping_tps = set()
        for tps in valid_timepoints_by_key.values():
            tps_set = set(tps)
            overlapping_tps |= (seen_tps & tps_set)
            seen_tps |= tps_set
        if overlapping_tps:
            failure_reasons.append(f'timepoints assigned to more than one sister/side segment: '
                                   f'{sorted(overlapping_tps)}')

        unexp_tps = set(df['timepoint'].values) - seen_tps
        if unexp_tps:
            failure_reasons.append(f'unexpected timepoints are detected: {unexp_tps}')

        if failure_reasons:
            raise ValueError(failure_reasons)

    def _build_combined_df(self) -> pd.DataFrame:
        """Concatenate the four segments' data into one dataframe, tagged by segment.

        Returns:
            pd.DataFrame: Rows from all four segments, sorted by timepoint,
                with `sister` and `side` columns identifying segment membership.
        """
        parts = []
        for (sister, side), segment in self._segments.items():
            part = segment.df.copy()
            if 'timepoint' not in part.columns:
                part = part.reset_index()
            part['sister'] = sister
            part['side'] = side
            parts.append(part)
        return pd.concat(parts).sort_values('timepoint').reset_index(drop=True)

    def _validate_sister_side(self, sister: str, side: str) -> None:
        """Validate `sister`/`side` selector arguments.

        Raises:
            ValueError: If `sister` is not one of "both", "sister1", "sister2",
                or `side` is not one of "both", "left", "right".
        """
        if sister != 'both' and sister not in self._SISTERS:
            raise ValueError('`sister` must be one of "both", "sister1", "sister2".')
        if side != 'both' and side not in self._SIDES:
            raise ValueError('`side` must be one of "both", "left", "right".')

    def _subset_df(self, sister: str = 'both', side: str = 'both') -> pd.DataFrame:
        self._validate_sister_side(sister, side)
        df = self.df
        if sister != 'both':
            df = df[df['sister'] == sister]
        if side != 'both':
            df = df[df['side'] == side]
        return df

    def get_gyration_radius(self, sister: str = 'both', side: str = 'both') -> float:
        """Radius of gyration for a sister/side subset of the trace.

        Args:
            sister: One of "both", "sister1", "sister2".
            side: One of "both", "left", "right".

        Returns:
            float: Radius of gyration, or `numpy.nan` if any selected
                coordinate is missing.
        """
        self._validate_sister_side(sister, side)
        if sister != 'both' and side != 'both':
            return self._segments[(sister, side)].gyration_radius
        subset_df = self._subset_df(sister, side)
        return Trace._compute_gyration_radius(subset_df[self._coord_cols].to_numpy())

    @property
    def distance_matrix(self) -> NDArray[np.float64]:
        """Pairwise Euclidean distance matrix across all four segments.

        Returns:
            np.ndarray: Symmetric (n x n) matrix of pairwise distances.
        """
        if self._distance_matrix is None:
            coords = self.df[self._coord_cols].to_numpy()
            self._distance_matrix = squareform(pdist(coords, metric='euclidean'))
        return self._distance_matrix

    def _sister_mask(self, sister: str) -> NDArray[np.bool_]:
        return (self.df['sister'] == sister).to_numpy()

    @property
    def sister1_dist_mtx(self) -> NDArray[np.float64]:
        """Pairwise distance matrix restricted to sister1 (both sides)."""
        mask = self._sister_mask('sister1')
        return self.distance_matrix[np.ix_(mask, mask)]

    @property
    def sister2_dist_mtx(self) -> NDArray[np.float64]:
        """Pairwise distance matrix restricted to sister2 (both sides)."""
        mask = self._sister_mask('sister2')
        return self.distance_matrix[np.ix_(mask, mask)]

    @property
    def trans_dist_mtx(self) -> NDArray[np.float64]:
        """Pairwise distances between sister1 and sister2, spanning the whole trace."""
        mask1 = self._sister_mask('sister1')
        mask2 = self._sister_mask('sister2')
        return self.distance_matrix[np.ix_(mask1, mask2)]

    @property
    def cut_end_dist_mtx(self) -> NDArray[np.float64]:
        """Pairwise distances between the four cut-adjacent endpoints.

        The four points are the timepoint nearest the cut site on each of the
        four segments: the last timepoint of each `left` segment and the
        first timepoint of each `right` segment.

        Returns:
            np.ndarray: Symmetric (4 x 4) matrix of pairwise distances, in the
                order sister1-left, sister1-right, sister2-left, sister2-right.
        """
        if self._cut_end_dist_mtx is None:
            keys = list(self._segments)
            coords = np.stack([getattr(self._segments[key], self._CUT_EDGE_ATTR[key[1]])
                               for key in keys])
            self._cut_end_dist_mtx = squareform(pdist(coords, metric='euclidean'))
        return self._cut_end_dist_mtx

    def get_distances(self, kind: str = 'all') -> NDArray[np.float64]:
        if kind == 'all':
            return self.distance_matrix
        elif kind == 'sister1':
            return self.sister1_dist_mtx
        elif kind == 'sister2':
            return self.sister2_dist_mtx
        elif kind == 'trans':
            return self.trans_dist_mtx
        elif kind == 'cut_ends':
            return self.cut_end_dist_mtx
        else:
            raise ValueError("kind must be 'all', 'sister1', 'sister2', 'trans', or 'cut_ends'")
