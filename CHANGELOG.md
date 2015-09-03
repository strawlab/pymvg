# v2.1.0 (unreleased)
---------------------

## Improvements

- Test against Python 3.4 using Travis

## API changes

- Properties on `CameraModel` (e.g. `cam.M`) are deprecated and will
  be removed in a future release.
- Intrinsic parameter matrix is normalized upon loading in
  `CameraModel._from_parts`, which is called by most constructors.

## Bugfixes

- Mirroring and flipping cameras with skew, distortion, and
  rectification work now.

2.0.0
-----

First real release
