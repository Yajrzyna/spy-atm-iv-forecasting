# Data (not included)

OptionsDX data is licensed, so it is not part of this repository. Place it here, or point `OPTIONSDX_DIR` at it:

```
30 minutes/SPY Option Chain/txt/    raw OptionsDX 30-minute SPY chains (used by the models)
End of Day/VIX Option Chains/txt/   raw OptionsDX end-of-day VIX chains (regime feature)
End of Day/SPY Option Chains/txt/   end-of-day SPY chains (optional, built by 01 only)
End of Day/SPX Option Chain/txt/    end-of-day SPX chains (optional, built by 01 only)
```

`01_build_parquets.py` writes the `parquet/` folders next to them; the later steps read those.
