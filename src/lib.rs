use gstreamer::glib;

mod filter;

/// Initializes the plugin by registering the filter.
///
/// # Arguments
///
/// * `plugin` - A reference to the GStreamer plugin.
///
/// # Returns
///
/// Returns `Ok(())` if the filter is successfully registered, otherwise returns a `glib::BoolError`.
fn plugin_init(plugin: &gstreamer::Plugin) -> Result<(), glib::BoolError> {
    filter::register(plugin)?;
    Ok(())
}

// This macro defines a GStreamer plugin named "whisper" with the given metadata.
// The metadata includes the plugin's description, initialization function, version, license, name, origin, repository, and build date.
gstreamer::plugin_define!(
    whisper,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MIT/Apache-2.0",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
