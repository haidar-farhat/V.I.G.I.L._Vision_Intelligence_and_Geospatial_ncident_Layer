# Map data

Operator-imported offline map packages live here. **Nothing in this directory is
committed**, and nothing is ever downloaded automatically.

A package is:

```
region.pmtiles      vector or raster tiles
style.json          references only local sources, glyphs and sprites
metadata.json       region, bounds, zoom range, tile type, version, checksum
```

Import validates CRS, bounds, zoom range, tile type, integrity, and that every
style dependency resolves inside the package. A style that resolves at import time
but reaches the network at render time is the exact failure that check exists for.

If no package is installed the application says so and offers the import flow. It
never substitutes an online source.

See [docs/MAPS.md](../docs/MAPS.md).
