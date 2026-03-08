from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationConfig:
    """Input validation and geometry-normalization options."""

    use_representative_points: bool = True
    point_precision: int = 6
    fix_invalid_geometries: bool = True


@dataclass(frozen=True)
class TrunkConfig:
    """Parameters for trunk generation and partitioning."""

    boundary_spacing: float = 100.0
    boundary_buffer: float | None = None
    boundary_buffer_ratio: float = 0.2
    subarea_spacing: float = 25.0
    length_removal: float = 200.0
    split_distance: float = 750.0

    def resolved_boundary_buffer(self) -> float:
        """Return the effective trunk boundary exclusion distance."""
        if self.boundary_buffer is not None:
            return self.boundary_buffer
        return max(1.0, self.boundary_spacing * self.boundary_buffer_ratio)


@dataclass(frozen=True)
class PoleConfig:
    """Parameters for lattice pole creation."""

    pole_spacing: float = 50.0
    trunk_buffer: float = 25.0
    point_precision: int = 6


@dataclass(frozen=True)
class RoutingConfig:
    """Parameters for sparse, biased secondary routing."""

    k_neighbors: int = 8
    neighbor_radius_factor: float = 2.2
    trunk_connection_radius_factor: float = 2.0
    angle_tolerance_degrees: float = 8.0
    diagonal_penalty: float = 2.0
    orthogonal_trunk_bonus: float = 0.5
    trunk_trunk_factor: float = 0.05
    trunk_distance_penalty_scale: float = 0.1
    include_trunk_trunk_in_secondary: bool = False


@dataclass(frozen=True)
class EngineConfig:
    """Full engine configuration."""

    validation: ValidationConfig = field(default_factory=ValidationConfig)
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    poles: PoleConfig = field(default_factory=PoleConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
