from pysc2.lib.features import *
from pysc2.lib import stopwatch

sw = stopwatch.sw

class ScreenFeatures(collections.namedtuple("ScreenFeatures", ["height_map", "visibility_map",
                    "creep", "power", "player_relative", "unit_type", "unit_density", "unit_density_aa"])):
  """The set of screen feature layers."""
  __slots__ = ()

  def __new__(cls, **kwargs):
    feats = {}
    for name, (scale, type_, palette, clip) in six.iteritems(kwargs):
      feats[name] = Feature(
          index=ScreenFeatures._fields.index(name),
          name=name,
          layer_set="renders",
          full_name="screen " + name,
          scale=scale,
          type=type_,
          palette=palette(scale) if callable(palette) else palette,
          clip=clip)
    return super(ScreenFeatures, cls).__new__(cls, **feats)


class MinimapFeatures(collections.namedtuple("MinimapFeatures", [
    "height_map", "visibility_map", "creep", "player_relative", "unit_type"])):
  """The set of minimap feature layers."""
  __slots__ = ()

  def __new__(cls, **kwargs):
    feats = {}
    for name, (scale, type_, palette) in six.iteritems(kwargs):
      feats[name] = Feature(
          index=MinimapFeatures._fields.index(name),
          name=name,
          layer_set="minimap_renders",
          full_name="minimap " + name,
          scale=scale,
          type=type_,
          palette=palette(scale) if callable(palette) else palette,
          clip=False)
    return super(MinimapFeatures, cls).__new__(cls, **feats)

SCREEN_FEATURES = ScreenFeatures(
    height_map=(256, FeatureType.SCALAR, colors.winter, False),
    visibility_map=(4, FeatureType.CATEGORICAL,
                    colors.VISIBILITY_PALETTE, False),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE, False),
    power=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
    player_relative=(5, FeatureType.CATEGORICAL,
                     colors.PLAYER_RELATIVE_PALETTE, False),
    unit_type=(1850, FeatureType.CATEGORICAL, colors.unit_type, False),
    unit_density=(16, FeatureType.SCALAR, colors.hot, False),
    unit_density_aa=(256, FeatureType.SCALAR, colors.hot, False),
)

MINIMAP_FEATURES = MinimapFeatures(
    height_map=(256, FeatureType.SCALAR, colors.winter),
    visibility_map=(4, FeatureType.CATEGORICAL, colors.VISIBILITY_PALETTE),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE),
    player_relative=(5, FeatureType.CATEGORICAL,
                     colors.PLAYER_RELATIVE_PALETTE),
    unit_type=(1850, FeatureType.CATEGORICAL, colors.unit_type)
)

class SpatialFeatures(Features):
    def observation_spec(self):
        """The observation spec for the SC2 environment.
        Returns:
          The dict of observation names to their tensor shapes. Shapes with a 0 can
          vary in length, for example the number of valid actions depends on which
          units you have selected.
        """
        return {
            "screen": (len(SCREEN_FEATURES),
                       self._screen_size_px.y,
                       self._screen_size_px.x),
            "minimap": (len(MINIMAP_FEATURES),
                        self._minimap_size_px.y,
                        self._minimap_size_px.x),
            "player": (11,),
            "score": (13,)
        }

    @sw.decorate
    def transform_obs(self, obs):
        """Render some SC2 observations into something an agent can handle."""
        out = {}

        with sw("feature_layers"):
            out["screen"] = np.stack(
                f.unpack(obs)/f.scale for f in SCREEN_FEATURES).astype(np.float32, copy=False)
            out["minimap"] = np.stack(
                f.unpack(obs)/f.scale for f in MINIMAP_FEATURES).astype(np.float32, copy=False)

        out["player"] = np.array([
            obs.game_loop - 1,
            obs.player_common.minerals,
            obs.player_common.vespene,
            obs.player_common.food_used,
            obs.player_common.food_cap,
            obs.player_common.food_army,
            obs.player_common.food_workers,
            obs.player_common.idle_worker_count,
            obs.player_common.army_count,
            obs.player_common.warp_gate_count,
            obs.player_common.larva_count,
        ], dtype=np.int32)

        out["score"] = np.array([
            obs.score.score,
            obs.score.score_details.idle_production_time,
            obs.score.score_details.idle_worker_time,
            obs.score.score_details.total_value_units,
            obs.score.score_details.total_value_structures,
            obs.score.score_details.killed_value_units,
            obs.score.score_details.killed_value_structures,
            obs.score.score_details.collected_minerals,
            obs.score.score_details.collected_vespene,
            obs.score.score_details.collection_rate_minerals,
            obs.score.score_details.collection_rate_vespene,
            obs.score.score_details.spent_minerals,
            obs.score.score_details.spent_vespene,
        ], dtype=np.int32)

        return out