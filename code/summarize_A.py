import glob
import json

for path in sorted(glob.glob("code/tsfm_group_*_h5.json")):
    d = json.load(open(path, encoding="utf-8"))
    b = d["schemes"]["binary"]["10bps"]
    print(
        d.get("norm", "?"),
        "| IC", d["ic_mean"],
        "| ICIR", d["icir"],
        "| gross", b["gross_sharpe"],
        "| net10", b["net_sharpe"],
        "| turnover", b["turnover_per_rebalance"],
        "| grossbps", b["daily_gross_bps"],
    )