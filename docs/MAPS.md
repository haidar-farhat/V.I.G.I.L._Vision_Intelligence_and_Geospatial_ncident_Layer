# Maps

> **Status:** architecture specified; rendering and package import are `PLANNED`.
> The geospatial model, ground projection, FOV footprints and zone geometry that
> the map renders are implemented and tested. See [STATUS.md](../STATUS.md).

## Why the map is a first-class part of the product

Events without location are a list. Events with location are a picture of what is
happening to a site. The map is where "three people crossed Restricted Zone A" and
"camera 08 cannot see the gate" become the same kind of fact.

## Stack

```
 React -> MapLibre GL -> local style.json -> PMTiles / local tiles (file://)
                              |
                              +- glyphs (bundled)     no network at any layer
                              +- sprites (bundled)
                              +- GeoJSON overlays: cameras . FOV wedges . zones
                                                   tracks . events . incidents
```

MapLibre GL with PMTiles is chosen because it is the only mainstream stack that is
genuinely offline-capable. Explicitly **not** used:

- Google Maps API
- Mapbox online APIs (`mapbox-gl` is on the lint denylist)
- OpenStreetMap tile endpoints
- Any cloud map service, CDN font, or remote glyph server

Fonts, sprites and glyphs ship inside the bundle. A style referencing a remote
glyph URL is a broken map on an isolated network, so the lint fails on
hard-coded external URLs anywhere in source.

## Offline map packages

An operator imports a local package:

```
region.pmtiles      vector or raster tiles
style.json          references only local sources, glyphs and sprites
metadata.json       region, bounds, zoom range, tile type, version, checksum
```

Import validates:

- **CRS** — EPSG:3857 tiles, WGS84 coordinates
- **Bounding box** — sane, and covering the configured site
- **Zoom range** — declared range matches what the archive contains
- **Tile type** — vector or raster, matching the style's expectations
- **Integrity** — SHA-256 over the archive
- **Style dependencies** — every source, glyph and sprite the style names resolves
  to something inside the package. A style that resolves at import time but
  reaches the network at render time is the exact failure this check exists for.

Managed in the UI: import, remove, validate, set default.

### When map data is missing

```
OFFLINE MAP DATA NOT INSTALLED
```

The system states the absence and offers the import flow. It **never** substitutes
an online source. Silently falling back to the Internet would break the platform's
only unconditional promise, and would do it invisibly.

Camera monitoring, events and incidents all continue working with no map
installed. The map is how you understand a site, not how the system works.

## Camera geospatial model

```
lat . lon . altitude . heading . pitch . roll . hfov . vfov . range . mountHeight
```

The operator places a camera by clicking the map, then drags to rotate and adjusts
the field of view directly on the wedge. Changes persist immediately.

### The FOV footprint is an annular sector

A camera tilted downward **cannot see the ground at its own mast**. The bottom of
its frame lands some distance out, and the top of the frame lands further out
still — or above the horizon, in which case the stated range bounds it.

So the rendered footprint is a ring segment between the near and far ground
distances, not a pie slice from the mast.

```
        camera                     pie slice (wrong)   annular sector (right)
          |                            /````\               ..---..
          |  blind foreground         /      \            .'       '.
          v                          /________\          (___________)
```

Drawing the pie slice would tell an operator the camera covers ground it is
physically blind to. That is precisely the assumption that gets a site burgled,
so the geometry package computes the real footprint and a test asserts the blind
foreground is excluded.

## Tracks on the map

Image-space detections are projected onto the ground plane using the camera pose,
which gives an approximate map position **with an explicit uncertainty**.

Ground range is `mountHeight / tan(depression)`, so the error grows
super-linearly as a ray flattens toward the horizon. A detection 20 m from a
camera may be located to within a metre; the same camera's detection near the
horizon is uncertain by tens of metres.

The map therefore renders an uncertainty ellipse, not a dot. **False precision is
treated as a defect**, and the uncertainty travels with the position through every
layer of the system — the tracker, the database schema, the API and the UI.

Where projection is impossible — no pose, or a ray above the horizon — the system
renders the camera's own position tagged `CAMERA_FALLBACK`, with an uncertainty
covering the whole field of view. The operator learns "something is happening at
this camera", which is true, without the map implying a precision that does not
exist.

Measured against simulated ground truth, the current pipeline achieves a mean
error of about 0.5 m with 100% of errors falling inside the stated uncertainty —
on flat ground with a perfectly known pose. Real sites have neither.

## Zones

Drawn on the map and stored in geographic coordinates, so a zone is meaningful
independent of any one camera. Kinds: polygon, rectangle, circle, corridor and
line (a tripwire, which has no interior).

Zones nest — site, perimeter, restricted area, gate, asset — and events inherit
contextual information from the hierarchy.

## Indoor sites

Site → building → floor → room share the same zone and event model with a local
coordinate frame instead of a geographic one, so indoor deployments reuse the
entire engine rather than needing a parallel implementation. The map switches
between outdoor and indoor views.
