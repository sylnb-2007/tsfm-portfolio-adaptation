import numpy as np
import pandas as pd

za = np.load("code/tsfm_mv_h5_pred.npz")
zb = np.load("code/tsfm_uv_h5_pred.npz")
mv = pd.DataFrame(za["values"], index=pd.to_datetime(za["dates"]), columns=za["columns"])
uv = pd.DataFrame(zb["values"], index=pd.to_datetime(zb["dates"]), columns=zb["columns"])
common = mv.index.intersection(uv.index)
diff = (mv.loc[common] - uv.loc[common]).to_numpy()
print("common_dates", len(common))
print("max_abs_diff", float(np.nanmax(np.abs(diff))))
print("mean_abs_diff", float(np.nanmean(np.abs(diff))))
print("mean_pred_mv", float(np.nanmean(mv.loc[common].to_numpy())))
print("mean_pred_uv", float(np.nanmean(uv.loc[common].to_numpy())))
