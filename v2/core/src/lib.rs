//! VIGIL v2 engine core.
//!
//! The arithmetic that runs per pixel, per detection and per frame. Python
//! above it owns policy — what a track means, when an event is raised — and
//! this crate owns the numbers, because the numbers are where the time goes.
//!
//! Four kernels, chosen by measurement rather than by taste:
//!
//!  - [`camera`]: the pinhole model. A projection is cheap on its own; the
//!    orthophoto lattice asks for a thousand of them per frame per camera,
//!    and the covariance behind each one is a finite-difference Jacobian, so
//!    one "projection" is really thirteen.
//!  - [`assign`]: rectangular linear assignment (Jonker–Volgenant). Tracking
//!    needs a globally optimal matching, not the greedy pass v1 and v2 both
//!    used; SciPy is not a dependency of this product and a Python O(n^3)
//!    solver at 15 fps is not one either.
//!  - [`track`]: constant-velocity Kalman predict/update and the cost matrix
//!    that feeds the assignment.
//!  - [`ortho`]: barycentric rasterisation of a ground grid and the streaming
//!    per-cell median behind it. v1 measured this at 79 ms per frame per
//!    camera in NumPy, which is why v1 sampled four frames a second.
//!
//! No dependencies. For an offline appliance the dependency list is part of
//! the attack surface, and everything here is arithmetic.

pub mod assign;
pub mod camera;
pub mod ffi;
pub mod geodesy;
pub mod lens;
pub mod ortho;
pub mod track;
pub mod triangulate;
