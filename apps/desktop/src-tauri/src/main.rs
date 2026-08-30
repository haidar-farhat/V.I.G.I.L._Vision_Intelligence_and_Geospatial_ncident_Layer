// Sentinel Vision desktop shell.
//
// The Rust side owns everything the browser must never touch: the OS keychain,
// the filesystem, the lifecycle of the local services, the system tray and native
// notifications. The React layer asks for these through explicit IPC commands and
// never receives a secret in return - only a handle, or a success.
//
// STATUS: scaffolded, never compiled. No Rust toolchain was available in the
// environment where this was written, so treat every function below as an
// unverified sketch of the intended boundary rather than working code. See
// STATUS.md.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use keyring::Entry;
use serde::{Deserialize, Serialize};

/// Keychain service name. All Sentinel credentials live under this namespace.
const KEYCHAIN_SERVICE: &str = "com.sentinelvision.desktop";

#[derive(Debug, Serialize, Deserialize)]
pub struct CredentialRef {
    /// Opaque handle stored in the database. Never the credential itself.
    pub reference: String,
}

/// Store a camera credential in the OS keychain.
///
/// Returns only a reference. The password is never echoed back, never logged, and
/// never written to the database - the database stores this handle and nothing
/// else. On Windows this is Credential Manager, on macOS the Keychain, on Linux
/// the Secret Service.
#[tauri::command]
fn store_credential(
    camera_id: String,
    username: String,
    password: String,
) -> Result<CredentialRef, String> {
    let reference = format!("camera/{camera_id}");

    let entry = Entry::new(KEYCHAIN_SERVICE, &reference)
        .map_err(|error| format!("keychain unavailable: {error}"))?;

    // Username and password are stored together so a rotation replaces both
    // atomically; a half-rotated credential is an outage.
    let payload = serde_json::json!({ "username": username, "password": password }).to_string();

    entry
        .set_password(&payload)
        .map_err(|error| format!("could not store credential: {error}"))?;

    Ok(CredentialRef { reference })
}

/// Remove a credential. Called when a camera is deleted, and audited by the API.
#[tauri::command]
fn delete_credential(reference: String) -> Result<(), String> {
    let entry = Entry::new(KEYCHAIN_SERVICE, &reference)
        .map_err(|error| format!("keychain unavailable: {error}"))?;

    match entry.delete_credential() {
        Ok(()) => Ok(()),
        // Already gone is the desired end state, not a failure.
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(format!("could not delete credential: {error}")),
    }
}

/// Whether a credential exists, without retrieving it.
///
/// The UI needs to show "credentials configured" without ever handling the
/// secret, so this deliberately returns a boolean rather than the value.
#[tauri::command]
fn has_credential(reference: String) -> bool {
    Entry::new(KEYCHAIN_SERVICE, &reference)
        .and_then(|entry| entry.get_password())
        .is_ok()
}

#[derive(Debug, Serialize)]
pub struct NetworkIsolation {
    pub wan: &'static str,
    pub lan: &'static str,
    pub internet_dependency: &'static str,
    pub cloud_services: &'static str,
    pub telemetry: &'static str,
}

/// The network isolation panel.
///
/// These values are constants rather than settings because they describe what the
/// application *is*, not how it is configured. There is no cloud client to
/// disable and no telemetry sink to opt out of.
#[tauri::command]
fn network_isolation() -> NetworkIsolation {
    NetworkIsolation {
        wan: "BLOCKED / NOT REQUIRED",
        lan: "ACTIVE",
        internet_dependency: "NONE",
        cloud_services: "DISABLED",
        telemetry: "DISABLED",
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_notification::init())
        .invoke_handler(tauri::generate_handler![
            store_credential,
            delete_credential,
            has_credential,
            network_isolation
        ])
        .run(tauri::generate_context!())
        .expect("failed to start Sentinel Vision");
}
