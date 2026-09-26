Presets are packaged in `src/ksvd/presets/{smoke,paper}.json` so installed
command-line tools do not depend on the current working directory.

Place JSON overrides here. For example:

```json
{"ks": [2], "mus": [0.1], "max_steps": 1200, "seeds": [3, 4, 5]}
```

Pass `--config configs/my_trace.json` to a trace study. Overrides merge into
the selected preset; the complete effective configuration is saved in the
manifest. `paper` is opt-in and can take substantial time. No paper sweep
is run by unit tests or smoke tests.
