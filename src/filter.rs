mod imp;

use gstreamer::{glib, prelude::StaticType, Rank};

// A GStreamer filter element that applies "whisper" AI to an audio stream.
glib::wrapper! {
  pub struct WhisperFilter(ObjectSubclass<imp::WhisperFilter>) @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
}

/// Registers the whisper filter plugin with GStreamer.
///
/// # Arguments
///
/// * `plugin` - A reference to the GStreamer plugin.
///
/// # Returns
///
/// * `Result<(), glib::BoolError>` - A result indicating success or failure.
pub fn register(plugin: &gstreamer::Plugin) -> Result<(), glib::BoolError> {
    gstreamer::Element::register(
        Some(plugin),
        "whisper",
        Rank::None,
        WhisperFilter::static_type(),
    )
}
