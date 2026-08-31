//! Sentinel Vision engine core.
//!
//! The hot path: geometry, projection, zones and tracking. Runs per detection,
//! per frame, per camera, so it lives in Rust rather than Python — and behind a
//! plain C ABI rather than Python-specific bindings, so nothing above it is
//! welded to one runtime.
//!
//! No dependencies. For a security appliance the dependency list is part of the
//! attack surface, and everything here is arithmetic.

pub mod ffi;
pub mod geometry;
pub mod tracking;
