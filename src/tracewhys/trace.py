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
            Defaults to False: an imputed spot lies on the straight line between
            its neighbours, which shortens its distances to them and biases
            distance-based metrics.
        missing (str): How metrics treat coordinates that are still missing
            after imputation (if any). 'nan' (default): a metric that uses a
            missing spot is `numpy.nan`. 'skip': missing spots are left out and
            metrics are computed from the remaining spots (see `RnaTrace`).
        min_spots (int): In 'skip' mode, the minimum number of spots with a
            position (detected, or imputed when `impute=True`) needed for
            metrics computed over the spot cloud; below it they are `numpy.nan`.

    Attributes:
        original_timepoint_mask (np.ndarray): Boolean mask per timepoint indicating
            whether a coordinate was present in the original pre-imputation input.
        df (pd.DataFrame): Working dataframe (imputed if `impute=True`).
        dim (str): Coordinate dimensionality ('2d' or '3d').
        missing (str): Missing-spot mode, 'nan' or 'skip'.
        min_spots (int): Minimum number of spots for 'skip'-mode metrics.

    Notes:
        - Properties prefixed with `original_` refer to the pre-imputation state.
        - Metric properties return `numpy.nan` when required coordinates are missing
          (`missing='nan'`), or when too few spots remain (`missing='skip'`).
        - In 'skip' mode, a trace missing an end spot covers a shorter stretch of
          the molecule, so size-type metrics (radius of gyration, convex hull) are
          systematically smaller than for a complete trace. Pooling partial and
          complete traces mixes these populations.
        - Private helpers and cached attributes are named with a leading underscore and
          are implementation details (not part of the public API).
    """
    def __init__(self,
                 df: pd.DataFrame,
                 valid_timepoints: Sequence[int],
                 dim: Literal['2d', '3d'] = '2d',
                 impute: bool = False,
                 missing: Literal['nan', 'skip'] = 'nan',
                 min_spots: int = 3,
                ) -> None:
        """Create a trace run from coordinate measurements.

        Args:
            df: DataFrame containing timepoint-indexed coordinate measurements.
            valid_timepoints: Expected ordered timepoints for the trace.
            dim: Coordinate dimensionality, either ``"2d"`` or ``"3d"``.
            impute: Whether to linearly impute missing interior coordinates.
            missing: Missing-spot mode, either ``"nan"`` or ``"skip"``.
            min_spots: Minimum number of spots for ``"skip"``-mode metrics.

        Raises:
            ValueError: If ``dim`` or ``missing`` is invalid, ``min_spots`` is
                below 2, or required columns/timepoints are missing or unexpected.
        """
        if missing not in ('nan', 'skip'):
            raise ValueError(f'`missing` must be "nan" or "skip", but "{missing}" was provided.')
        if min_spots < 2:
            raise ValueError(f'`min_spots` must be at least 2, but {min_spots} was provided.')

        self._valid_timepoints: Sequence[int] = valid_timepoints
        self._coord_cols: list[str] = _coord_cols_for_dim(dim)
        self.dim: Literal['2d', '3d'] = dim
        self.missing: Literal['nan', 'skip'] = missing
        self.min_spots: int = min_spots

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
    def is_complete(self) -> bool:
        """Whether every timepoint was present in the original input.

        For a complete trace, metrics do not depend on `impute` or `missing`,
        so selecting complete traces after computing metrics in any mode gives
        the full-trace-only analysis.

        Returns:
            bool: True if no timepoint was missing before imputation.
        """
        return bool(self.original_timepoint_mask.all())

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

    @property
    def placed_timepoint_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of timepoints that have a position in the current `df`.

        Returns:
            np.ndarray: 1-D boolean array with length equal to the number of valid
                timepoints. True indicates the coordinate was detected or, when
                `impute=True`, imputed.
        """
        return ~np.isnan(self._coords).any(axis=1)

    @property
    def placed_count(self) -> int:
        """Number of timepoints that have a position in the current `df`.

        Returns:
            int: Count of detected spots, plus imputed spots when `impute=True`.
        """
        return int(self.placed_timepoint_mask.sum())

    @property
    def _metric_coords(self) -> NDArray[np.float64] | None:
        """Coordinates used by metrics computed over the whole spot cloud.

        Returns:
            np.ndarray or None: In 'nan' mode, all coordinates, or None if any is
                missing. In 'skip' mode, the coordinates of the placed spots, or
                None if there are fewer than `min_spots` of them.
        """
        if self.missing == 'skip':
            if self.placed_count < self.min_spots:
                return None
            return self._coords[self.placed_timepoint_mask]
        if self._has_remaining_missing_segments:
            return None
        return self._coords

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
                are missing from the current `df` ('nan' mode) or fewer than
                `min_spots` are placed ('skip' mode, computed over placed spots).
        """
        if self._gyration_radius is None:
            coords = self._metric_coords
            self._gyration_radius = np.nan if coords is None else self._compute_gyration_radius(coords)
        return float(self._gyration_radius)

    @property
    def distance_matrix(self) -> NDArray[np.float64]:
        """Pairwise Euclidean distance matrix between timepoints.

        Returns:
            np.ndarray: Symmetric (n x n) matrix of pairwise distances, aligned
                with the valid timepoints. Rows and columns of missing spots are
                `numpy.nan`.
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
            Defaults to False (see `Trace`).
        missing (str): Missing-spot mode, 'nan' (default) or 'skip' (see `Trace`).
        min_spots (int): Minimum number of spots for 'skip'-mode metrics.
        metadata (TraceMetadata | None): Optional metadata for the trace.

    Attributes:
        metadata (dict | None): Trace metadata if provided.

    Notes:
        - See `Trace` for the shared single-run behavior: imputation,
          `original_timepoint_mask`, dropout indicators (`original_missing_count`,
          `has_original_missing_left_edge`, ...), `gyration_radius`,
          `distance_matrix`, etc.
        - Metric properties return `numpy.nan` when required coordinates are missing.
        - With `missing='skip'`, each metric uses the spots it can:
            - `gyration_radius`, `asphericity` and the convex hull: the placed
              spots, if there are at least `min_spots` (the hull also needs 3
              non-collinear spots in 2D, 4 non-coplanar in 3D);
            - `scaling_exponent`: the pairs of placed spots, at their separation
              along the trace (not their rank among the placed spots);
            - `half_extent_asymmetry`: the placed spots of each half, with halves
              still defined by position along the trace; needs at least 2 placed
              spots per half;
            - `end_to_end_distance` and `contour_length` stay `numpy.nan` when an
              end spot (respectively, any spot) is missing, since skipping a spot
              would shorten them.
        - To restrict an analysis to full traces, select the rows of
          `get_all_metrics()` whose `complete` column is True (see `is_complete`).
    """
    def __init__(self,
                 df: pd.DataFrame,
                 valid_timepoints: Sequence[int],
                 dim: Literal['2d', '3d'] = '2d',
                 impute: bool = False,
                 metadata: dict | None = None,
                 missing: Literal['nan', 'skip'] = 'nan',
                 min_spots: int = 3,
                ) -> None:
        """Create an RNA trace from coordinate measurements.

        Args:
            df: DataFrame containing timepoint-indexed coordinate measurements.
            valid_timepoints: Expected ordered timepoints for the trace.
            dim: Coordinate dimensionality, either ``"2d"`` or ``"3d"``.
            impute: Whether to linearly impute missing interior coordinates.
            metadata: Optional trace metadata.
            missing: Missing-spot mode, either ``"nan"`` or ``"skip"``.
            min_spots: Minimum number of spots for ``"skip"``-mode metrics.

        Raises:
            ValueError: If ``dim`` or ``missing`` is invalid, ``min_spots`` is
                below 2, or required columns/timepoints are missing or unexpected.
        """
        super().__init__(df, valid_timepoints, dim=dim, impute=impute,
                         missing=missing, min_spots=min_spots)
        self.metadata: dict | None = metadata
        self._convex_hull: ConvexHull | None = None

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
        negative values indicate the left half is more extended. "Left" is the
        half with the earlier valid timepoints; whether that is the 5' or the
        3' half depends on the probe design. Values are
        typically in the range (-1, 1) when radii are positive and finite.

        In 'skip' mode the halves are still split by position along the trace,
        and each Rg is computed over the placed spots of its half.

        Returns:
            float: The half-extent asymmetry, or ``numpy.nan`` if the metric
                cannot be computed because of missing coordinates ('nan' mode)
                or because a half has fewer than 2 placed spots ('skip' mode).
                If both radii are zero the result is undefined.
        """
        if self.missing == 'nan' and self._has_remaining_missing_segments:
            return np.nan
        coords = self._coords
        N = coords.shape[0]
        halves = [coords[:N // 2, :], coords[N - N // 2:, :]]
        if self.missing == 'skip':
            halves = [half[~np.isnan(half).any(axis=1)] for half in halves]
            if any(len(half) < 2 for half in halves):
                return np.nan
        rg_left, rg_right = (self._compute_gyration_radius(half) for half in halves)
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
                are complete ('nan' mode) or at least `min_spots` are placed
                ('skip' mode, hull of the placed spots); None otherwise, or if
                the spots are degenerate (e.g. collinear in 2D).
        """
        coords = self._metric_coords
        if coords is None:
            return None
        if self._convex_hull is None:
            try:
                self._convex_hull = ConvexHull(coords)
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
                current trace is missing ('nan' mode) or fewer than `min_spots`
                are placed ('skip' mode, computed over placed spots).
        """
        c = self._metric_coords
        if c is None:
            return np.nan
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
                "anchor_5": Distances from the first valid timepoint
                    (index 0) to each later timepoint.
                "anchor_3": Distances from the last valid timepoint
                    (last index) to each earlier timepoint.
                "all": All pairwise distances grouped by genomic/timepoint
                    separation.
                "agg": One aggregated distance per separation, using
                    ``agg_func``.
            agg_func: Aggregation function used when ``mode`` is ``"agg"``.
                Defaults to ``np.nanmedian``.

        Returns:
            pd.DataFrame: DataFrame with columns ``separation``, ``dist``,
                ``n_pairs`` and ``n_missing``. ``n_pairs`` is the total number of
                pairs at that separation, regardless of missingness (1 per row in
                the non-aggregated modes). ``n_missing`` is the number of those
                pairs in which at least one segment was missing in the original
                input (before imputation).

        Note:
            The mode names assume the first valid timepoint is the 5' end.
            Which end it is depends on the probe design (the order in which
            segments are imaged); when it is the 3' end, "anchor_5" anchors
            at the 3' end and "anchor_3" at the 5' end.
        """
        mtx = self.distance_matrix
        n = mtx.shape[0]
        orig_missing = ~self.original_timepoint_mask

        if mode == 'anchor_5':
            records = [{'separation': sep,
                        'dist': mtx[0, sep],
                        'n_pairs': 1,
                        'n_missing': int(orig_missing[0] | orig_missing[sep])}
                       for sep in range(1, n)]
            df = pd.DataFrame.from_records(records)

        elif mode == 'anchor_3':
            records = [{'separation': sep,
                        'dist': mtx[-1, n - 1 - sep],
                        'n_pairs': 1,
                        'n_missing': int(orig_missing[-1] | orig_missing[n - 1 - sep])}
                       for sep in range(1, n)]
            df = pd.DataFrame.from_records(records)

        elif mode in ('all', 'agg'):
            df = pd.concat([pd.DataFrame({'separation': sep,
                                          'dist': mtx.diagonal(sep),
                                          'n_pairs': 1,
                                          'n_missing': orig_missing[:-sep] | orig_missing[sep:]})
                            for sep in range(1, n)],
                           ignore_index=True)
            df['n_missing'] = df['n_missing'].astype(int)

            if mode == 'agg':
                df = (df.groupby('separation', as_index=False)
                        .agg({'dist': agg_func or np.nanmedian,
                              'n_pairs': 'sum',
                              'n_missing': 'sum'}))
        else:
            raise ValueError(f'`mode` must be one of "anchor_5", "anchor_3", "all", "agg", '
                             f'but "{mode}" was provided.')
        return df

    @property
    def scaling_exponent(self) -> float:
        """Slope of log10(distance) against log10(separation) over all pairs.

        In 'skip' mode, only pairs whose two spots are placed are used, each at
        its separation along the trace.

        Returns:
            float: The fitted slope, or ``numpy.nan`` if any distance is missing
                ('nan' mode), or if fewer than `min_spots` spots are placed or
                fewer than 2 separations remain ('skip' mode).
        """
        scaling_df = self.get_scaling_table(mode='all')
        if self.missing == 'skip':
            scaling_df = scaling_df.dropna(subset=['dist'])
            if self.placed_count < self.min_spots or scaling_df['separation'].nunique() < 2:
                return np.nan
        return float(stats.linregress(np.log10(scaling_df['separation']),
                                      np.log10(scaling_df['dist']))
                          .slope)

    def get_all_metrics(self):
        """Compute and return all available metrics for the trace."""
        metrics = dict(Rg=self.gyration_radius,
                       asphericity=self.asphericity,
                       end_to_end_dist=self.end_to_end_distance,
                       contour_length=self.contour_length,
                       end_to_end_dist_norm=self.end_to_end_distance / self.contour_length,
                       half_ext_asym=self.half_extent_asymmetry,
                       scaling_exp=self.scaling_exponent,
                       perimeter=self.perimeter,
                       area=self.area,
                       volume=self.volume)

        dropouts_info = dict(complete=self.is_complete,
                             missing_count=self.original_missing_count,
                             present_count=self.original_present_count,
                             placed_count=self.placed_count,
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

    Composed of four `Trace` subtraces: sister1/sister2, each split into a
    left and right side relative to the DNA-repair cut site. Every subtrace
    has its own set of valid timepoints, and missing coordinates are imputed
    only within a subtrace -- never across the cut site or between sisters.

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
            within each subtrace.
            Defaults to False (see `Trace`).
        metadata (dict | None): Optional metadata for the trace.

    Attributes:
        df (pd.DataFrame): Combined dataframe (concatenation of the four
            subtraces' data, sorted by timepoint), with `sister` and `side`
            columns tagging each row and an `original_present` column
            marking rows present before imputation.
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
                 impute: bool = False,
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

        self._subtraces: dict[tuple[str, str], Trace] = {
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
        """Validate the input dataframe and the four timepoint subtraces.

        Raises:
            ValueError: If required columns are missing, a timepoint is
                assigned to more than one sister/side subtrace, or `df`
                contains timepoints outside all four subtraces.
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
            failure_reasons.append(f'timepoints assigned to more than one sister/side subtrace: '
                                   f'{sorted(overlapping_tps)}')

        unexp_tps = set(df['timepoint'].values) - seen_tps
        if unexp_tps:
            failure_reasons.append(f'unexpected timepoints are detected: {unexp_tps}')

        if failure_reasons:
            raise ValueError(failure_reasons)

    def _build_combined_df(self) -> pd.DataFrame:
        """Concatenate the four subtraces' data into one dataframe, tagged by subtrace.

        Returns:
            pd.DataFrame: Rows from all four subtraces, sorted by timepoint,
                with `sister` and `side` columns identifying subtrace membership
                and an `original_present` column holding each subtrace's
                `original_timepoint_mask`.
        """
        parts = []
        for (sister, side), subtrace in self._subtraces.items():
            part = subtrace.df.copy()
            if 'timepoint' not in part.columns:
                part = part.reset_index()
            part['sister'] = sister
            part['side'] = side
            part['original_present'] = subtrace.original_timepoint_mask
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

    def _select_subtraces(self, sister: str = 'both', side: str = 'both') -> list[Trace]:
        """Return the subtraces matching a `sister`/`side` selection.

        Raises:
            ValueError: If `sister` or `side` is not a valid selector.
        """
        self._validate_sister_side(sister, side)
        return [subtrace for (sub_sister, sub_side), subtrace in self._subtraces.items()
                if sister in ('both', sub_sister) and side in ('both', sub_side)]

    @property
    def original_timepoint_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of the original timepoint presence before imputation.

        Returns:
            np.ndarray: 1-D boolean array aligned with the rows of `df` (and
                therefore with `distance_matrix`). True indicates the coordinate
                was present in the original pre-imputation input.
        """
        return self.df['original_present'].to_numpy(dtype=bool)

    def get_original_timepoint_mask(self, sister: str = 'both', side: str = 'both') -> NDArray[np.bool_]:
        """Original timepoint-presence mask for a sister/side subset of the trace.

        Args:
            sister: One of "both", "sister1", "sister2".
            side: One of "both", "left", "right".

        Returns:
            np.ndarray: 1-D boolean array aligned with the rows of the selected
                subset of `df`, sorted by timepoint.
        """
        return self._subset_df(sister, side)['original_present'].to_numpy(dtype=bool)

    @property
    def original_present_count(self) -> int:
        """Number of timepoints present in the original input, over all four subtraces."""
        return self.get_original_present_count()

    @property
    def original_missing_count(self) -> int:
        """Number of timepoints missing in the original input, over all four subtraces."""
        return self.get_original_missing_count()

    def get_original_present_count(self, sister: str = 'both', side: str = 'both') -> int:
        """Number of originally present timepoints in a sister/side subset.

        Args:
            sister: One of "both", "sister1", "sister2".
            side: One of "both", "left", "right".

        Returns:
            int: Count of timepoints that had coordinate data before imputation.
        """
        return sum(subtrace.original_present_count
                   for subtrace in self._select_subtraces(sister, side))

    def get_original_missing_count(self, sister: str = 'both', side: str = 'both') -> int:
        """Number of originally missing timepoints in a sister/side subset.

        Args:
            sister: One of "both", "sister1", "sister2".
            side: One of "both", "left", "right".

        Returns:
            int: Count of timepoints that were missing before imputation.
        """
        return sum(subtrace.original_missing_count
                   for subtrace in self._select_subtraces(sister, side))

    @property
    def has_original_consecutive_missing_timepoints(self) -> bool:
        """Whether any subtrace contains consecutive missing timepoints in the original input.

        Adjacency is evaluated within each subtrace only -- never across the cut
        site or between sisters.

        Returns:
            bool: True if at least one subtrace has a pair of adjacent timepoints
                both missing before imputation, False otherwise.
        """
        return any(subtrace.has_original_consecutive_missing_timepoints
                   for subtrace in self._subtraces.values())

    @property
    def has_remaining_missing_segments(self) -> bool:
        """Whether any coordinate in any subtrace remains missing after imputation."""
        return self.get_has_remaining_missing()

    @property
    def has_remaining_missing_edges(self) -> bool:
        """Whether any subtrace's first or last coordinate remains missing after imputation."""
        return self.get_has_remaining_missing(edges_only=True)

    def get_has_remaining_missing(self,
                                  sister: str = 'both',
                                  side: str = 'both',
                                  edges_only: bool = False,
                                 ) -> bool:
        """Whether coordinates in a sister/side subset remain missing after imputation.

        Args:
            sister: One of "both", "sister1", "sister2".
            side: One of "both", "left", "right".
            edges_only: If True, only check each selected subtrace's first and
                last coordinates; otherwise check all coordinates.

        Returns:
            bool: True if any selected subtrace still contains NaN coordinates
                (at its edges, if `edges_only`), False otherwise.
        """
        subtraces = self._select_subtraces(sister, side)
        if edges_only:
            return any(subtrace._has_remaining_missing_edges for subtrace in subtraces)
        return any(subtrace._has_remaining_missing_segments for subtrace in subtraces)

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
            return self._subtraces[(sister, side)].gyration_radius
        subset_df = self._subset_df(sister, side)
        return Trace._compute_gyration_radius(subset_df[self._coord_cols].to_numpy())

    @property
    def distance_matrix(self) -> NDArray[np.float64]:
        """Pairwise Euclidean distance matrix across all four subtraces.

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
        four subtraces: the last timepoint of each `left` subtrace and the
        first timepoint of each `right` subtrace.

        Returns:
            np.ndarray: Symmetric (4 x 4) matrix of pairwise distances, in the
                order sister1-left, sister1-right, sister2-left, sister2-right.
        """
        if self._cut_end_dist_mtx is None:
            keys = list(self._subtraces)
            coords = np.stack([getattr(self._subtraces[key], self._CUT_EDGE_ATTR[key[1]])
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
