from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class RnaExpConfig:
    """Configuration for a trace-expression experiment dataset.

    This object reads the trace table and group-timepoint metadata needed to
    validate the experiment layout.

    Attributes:
        table_path (str): CSV with a header row and comma-separated values. It
            contains one row per trace observation and is expected to include the
            columns ``trace_group``, ``timepoint``, ``traceId``, ``is_nuclear``,
            ``x``, ``y``, and ``z``.
        groupings_path (str): CSV with a header row and comma-separated values.
            It contains one row per trace group and must include the columns
            ``trace_group`` and ``expected_timepoints``. The
            ``expected_timepoints`` field stores comma-separated integers such as
            ``"1,2,3,4"``.
        groups_to_names (dict[str, str]): Mapping from trace-group labels to
            gene names used to annotate rows in the parsed output table.
        metadata (dict | None): Optional experiment-level metadata stored
            alongside the configuration.

    Notes:
        - Rows whose ``trace_group`` is not present in ``groups_to_names`` are
        discarded.
        - Missing specified groups in ``groupings_path`` raise ``ValueError``.
        - ``x``, ``y``, and ``z`` may be stored as plain numeric
          strings or strings with trailing annotation text; in the latter case, the
          first numeric token is extracted before conversion.
        - Rows with missing or malformed coordinate values are dropped.
    """

    table_path: str
    groupings_path: str
    groups_to_names: dict[str, str]
    metadata: dict | None = None
    _parsed_table: pd.DataFrame | None = field(default=None,
                                               init=False,
                                               repr=False,
                                               compare=False)
    _valid_group_timepoints: dict[str, tuple[int]] | None = field(default=None,
                                                                  init=False,
                                                                  repr=False,
                                                                  compare=False)

    def __post_init__(self) -> None:
        """Convert paths to Path objects and validate their existence."""
        self.table_path = Path(self.table_path)
        self.groupings_path = Path(self.groupings_path)

        self._validate_path(self.table_path, "table_path")
        self._validate_path(self.groupings_path, "groupings_path")

    @staticmethod
    def _validate_path(path: Path, field: str) -> None:
        """Validate that a given path exists and is a file.

        Args:
            path (Path): The path to validate.
            field (str): The name of the path field (for error messages).

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path is not a file.
        """
        if not path.exists():
            raise FileNotFoundError(f"{field} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{field} is not a file: {path}")

    @property
    def valid_group_timepoints(self) -> dict[str, tuple[int]]:
        """Expected timepoints for each specified trace group.

        Returns:
            dict[str, tuple[int]]: Mapping from group name to the valid
                timepoints expected for that group.
        """
        if self._valid_group_timepoints is None:
            self._valid_group_timepoints = self._parse_groupings()
        return self._valid_group_timepoints

    @property
    def valid_gene_timepoints(self) -> dict[str, tuple[int]]:
        """Expected timepoints for each gene name derived from the trace groups.

        Returns:
            dict[str, tuple[int]]: Mapping from gene name to the valid
                timepoints expected for that gene.
        """
        return {self.groups_to_names[group]: tps
                for group, tps in self.valid_group_timepoints.items()}

    def _parse_groupings(self) -> dict[str, tuple[int]]:
        """Load and validate the group-to-timepoint definitions.

        The groupings CSV is expected to have a header row, comma separators,
        and the columns ``trace_group`` and ``expected_timepoints``.

        Returns:
            dict[str, tuple[int]]: Mapping from each specified trace group to
                the valid timepoints declared in the CSV.

        Raises:
            ValueError: If any specified group is missing from the groupings
                file.
        """
        req_cols = ('trace_group', 'expected_timepoints')
        groupings = pd.read_csv(self.groupings_path,
                                header=0,
                                sep=',',
                                usecols=req_cols,
                                converters={'trace_group': str,
                                            'expected_timepoints': self._parse_timepoints})

        specified_groups = set(self.groups_to_names)
        available_groups = set(groupings['trace_group'])
        missing_groups = specified_groups - available_groups

        if missing_groups:
            raise ValueError(f"Specified trace groups missing from groupings file: {sorted(missing_groups)}.")

        valid_group_timepoints = groupings.set_index('trace_group')\
                                          ['expected_timepoints']\
                                          .loc[list(specified_groups)]\
                                          .to_dict()
        return valid_group_timepoints

    @staticmethod
    def _parse_timepoints(timepoints_str) -> tuple[int]:
        """Parse the comma-separated ``expected_timepoints`` field.

        The field is expected to contain values such as ``"1,2,3,4"`` after the
        CSV header has already been consumed by ``pandas.read_csv``.

        Args:
            timepoints_str: String containing comma-separated integer labels.

        Returns:
            tuple[int]: Parsed integer timepoints in their original sequence.
        """
        return tuple(int(tp.strip()) for tp in timepoints_str.split(','))

    @property
    def _valid_group_timepoints_df(self) -> pd.DataFrame:
        """Flatten the valid group-timepoint mapping into a long-form dataframe.

        Returns:
            pd.DataFrame: DataFrame with columns ``trace_group`` and ``timepoint``.
        """
        return pd.DataFrame([(group, tp)
                             for group, tps in self.valid_group_timepoints.items()
                             for tp in tps],
                            columns=['trace_group', 'timepoint'])

    @property
    def table(self) -> pd.DataFrame:
        """Parsed trace table with valid group/timepoint rows only.

        Adds a ``gene_name`` column based on the provided group-to-name mapping.

        Returns:
            pd.DataFrame: Parsed table for all specified trace groups and valid
                expected timepoints.
        """
        if self._parsed_table is None:
            self._parsed_table = self._parse_table()
        return self._parsed_table

    def _parse_table(self) -> pd.DataFrame:
        """Read, validate, and normalize the trace table for this experiment.

        The table CSV is expected to start with a header row and use comma
        separators. Required columns are ``trace_group``, ``timepoint``,
        ``traceId``, ``is_nuclear``, ``x``, ``y``, and ``z``. Coordinate values
        are coerced to numeric form when possible, rows whose ``trace_group`` is
        missing from ``groups_to_names`` are dropped, and the remaining rows are
        intersected with the valid timepoints declared for each group.

        Returns:
            pd.DataFrame: Parsed table containing only valid trace groups and
                valid group/timepoint pairs.
        """
        req_cols = ('trace_group', 'timepoint', 'traceId', 'is_nuclear', 'x', 'y', 'z')
        table = pd.read_csv(self.table_path,
                            header=0,
                            sep=',',
                            usecols=req_cols)

        for coord in ('x', 'y', 'z'):
            if not is_numeric_dtype(table[coord]):
                table[coord] = pd.to_numeric(table[coord].astype('string').str.split().str[0], errors='coerce')

        table['gene_name'] = table['trace_group'].map(self.groups_to_names)
        table = table.dropna(subset=['gene_name', 'x', 'y', 'z'], how='any')

        table = pd.merge(table,
                         self._valid_group_timepoints_df,
                         on=('trace_group', 'timepoint'),
                         how='inner')\
                  .reset_index(drop=True)
        return table


def write_metrics(metrics_df: pd.DataFrame, output_path: str) -> None:
    """Write the metrics dataframe to a parquet file.

    Args:
        metrics_df: DataFrame containing computed metrics for traces.
        output_path: Path to the output parquet file.
    """
    metrics_df.assign(distance_matrix=lambda df: df['distance_matrix'].map(lambda x: x.tolist()))\
              .to_parquet(output_path, engine='pyarrow', index=False)


def read_metrics(input_path: str) -> pd.DataFrame:
    """Read the trace metrics dataframe from a parquet file.

    Args:
        input_path: Path to the input parquet file.
    Returns:
        DataFrame containing the metrics.
    """
    metrics_df = pd.read_parquet(input_path)\
                   .assign(distance_matrix=lambda df: df['distance_matrix'].map(np.stack),
                           timepoint_mask=lambda df: df['timepoint_mask'].map(np.asarray))
    return metrics_df
