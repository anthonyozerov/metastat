# metastat

Minimal Python helpers for common meta-analysis intervals. Functions operate directly on NumPy arrays of `values` and `sigmas`.

Modules:
- `metastat.infer`: interval estimators returning `MetaResult`
- `metastat.test`: related statistical tests (e.g., `sign_rank_test`)
- `metastat.stat`: summary statistics (e.g., `I2`)
- `metastat.asym`: meta-analysis when studies have asymmetric errors

Examples:

```python
import numpy as np
import metastat

values = np.array([1.1, 0.8, 1.4])
sigmas = np.array([0.2, 0.25, 0.15])

res_fe = metastat.infer.fixed_effect(values, sigmas, coverage=0.95)
print(res_fe.interval, res_fe.point_est, res_fe.sigma)

res_re = metastat.infer.random_effects_dl(values, sigmas, coverage=0.95)
print(res_re.interval, res_re.point_est, res_re.tau, res_re.sigma)
```

Optional nonparametric sign-rank functions use `Rmath4` when installed.