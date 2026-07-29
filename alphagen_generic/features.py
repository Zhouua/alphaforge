from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType


high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open_ = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
vwap = Feature(FeatureType.VWAP)

# Signal at t is traded at open[t+1] and held through open[t+11].
# Keep this expression synchronized with experiment_protocol.QLIB_TARGET.
target = Ref(open_, -11) / Ref(open_, -1) - 1
