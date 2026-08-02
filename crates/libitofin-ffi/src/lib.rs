//! Minimal C ABI for embedding the libitofin core.
//!
//! The wrapper intentionally exposes metadata and stable error-code helpers
//! only. Pricing logic remains in `libitofin`; higher-level C functions (and a
//! `cbindgen` header) are tracked in `docs/oracle-coverage.md` under P2 and
//! can be added without creating a second implementation of the Rust API.

use std::ffi::c_char;

const VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
const ERROR_MESSAGES: [&[u8]; 3] = [b"success\0", b"invalid argument\0", b"calculation error\0"];

/// Returns the package version as a nul-terminated UTF-8 string.
///
/// The returned pointer is valid for the lifetime of the process and must not
/// be freed by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn libitofin_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

/// Returns a stable message for a libitofin FFI error code.
///
/// Unknown codes return `"calculation error"`. The returned pointer is
/// process-owned and must not be freed by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn libitofin_error_message(code: i32) -> *const c_char {
    let index = usize::try_from(code).unwrap_or(2);
    ERROR_MESSAGES
        .get(index)
        .unwrap_or(&ERROR_MESSAGES[2])
        .as_ptr()
        .cast()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn ffi_exports_basic_metadata_and_errors() {
        let version = unsafe { CStr::from_ptr(libitofin_version()) };
        assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));

        let success = unsafe { CStr::from_ptr(libitofin_error_message(0)) };
        assert_eq!(success.to_str().unwrap(), "success");

        let unknown = unsafe { CStr::from_ptr(libitofin_error_message(99)) };
        assert_eq!(unknown.to_str().unwrap(), "calculation error");
    }
}
