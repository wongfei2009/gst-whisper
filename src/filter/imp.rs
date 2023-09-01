mod vad;

use std::{env, sync::Mutex, time::Instant};

use byte_slice_cast::AsSliceOf;
use gstreamer::{
    element_imp_error,
    glib::{self, ParamSpec, Value},
    param_spec::GstParamSpecBuilderExt,
    prelude::{ParamSpecBuilderExt, ToValue},
    subclass::{
        prelude::{ElementImpl, GstObjectImpl, ObjectImpl, ObjectSubclass, ObjectSubclassExt},
        ElementMetadata,
    },
    traits::{ElementExt, PadExt},
    Buffer, Caps, CapsIntersectMode, ClockTime, CoreError, DebugCategory, ErrorMessage, EventView,
    FlowError, PadDirection, PadPresence, PadTemplate,
};
use gstreamer_audio::{AudioCapsBuilder, AudioLayout, AUDIO_FORMAT_S16};
use gstreamer_base::{
    subclass::{
        base_transform::{BaseTransformImpl, BaseTransformImplExt, GenerateOutputSuccess},
        BaseTransformMode,
    },
    BaseTransform,
};
use once_cell::sync::Lazy;
use webrtc_vad::VadMode;
use whisper_rs::{
    convert_integer_to_float_audio, FullParams, SamplingStrategy, WhisperContext, WhisperState,
};

const SAMPLE_RATE: usize = 16_000;
const DEFAULT_USE_VAD: bool = true;
const DEFAULT_VAD_MODE: &str = "quality";
const DEFAULT_MIN_VOICE_ACTIVITY_MS: u64 = 200;
const DEFAULT_LANGUAGE: &str = "en";
const DEFAULT_TRANSLATE: bool = false;
const DEFAULT_CONTEXT: bool = true;
const DEFAULT_LENGTH_MS: u64 = 10000;

/// A static variable that holds a lazy-initialized `WhisperContext` instance.
/// The `WhisperContext` instance is created using the path specified in the `WHISPER_MODEL_PATH` environment variable.
static WHISPER_CONTEXT: Lazy<WhisperContext> = Lazy::new(|| {
    let path = env::var("WHISPER_MODEL_PATH").unwrap();
    WhisperContext::new(&path).unwrap()
});

/// This static variable represents a lazy-initialized `DebugCategory` instance
/// for the "whisper" category. It is used for logging purposes in the `SpeechToTextFilter`
/// implementation, which uses the Whisper speech recognition engine.
static CAT: Lazy<DebugCategory> = Lazy::new(|| {
    DebugCategory::new(
        "whisper",
        gstreamer::DebugColorFlags::empty(),
        Some("Speech to text filter using Whisper"),
    )
});

static SINK_CAPS: Lazy<Caps> = Lazy::new(|| {
    AudioCapsBuilder::new()
        .format(AUDIO_FORMAT_S16)
        .layout(AudioLayout::NonInterleaved)
        .rate(SAMPLE_RATE as i32)
        .channels(1)
        .build()
});

static SRC_CAPS: Lazy<Caps> =
    Lazy::new(|| Caps::builder("text/x-raw").field("format", "utf8").build());

/// Struct representing the settings for the whisper filter.
struct Settings {
    use_vad: bool,
    vad_mode: String,
    min_voice_activity_ms: u64,
    language: String,
    translate: bool,
    context: bool,
}

/// Struct representing the state of the whisper filter.
struct State {
    /// The state of the whisper filter.
    whisper_state: WhisperState<'static>,
    /// The voice activity detector used to detect speech.
    voice_activity_detector: Option<vad::VoiceActivityDetector>,
    /// The current audio chunk being processed.
    chunk: Option<Chunk>,
    /// The previous audio buffer.
    prev_buffer: Vec<i16>,
}

/// A struct representing an audio chunk with its starting presentation timestamp and buffer.
struct Chunk {
    start_pts: ClockTime,
    buffer: Vec<i16>,
}

/// A struct representing a WhisperFilter.
pub struct WhisperFilter {
    #[allow(dead_code)]
    /// A mutex-protected struct containing the filter's settings.
    settings: Mutex<Settings>,
    /// A mutex-protected option containing the filter's state.
    state: Mutex<Option<State>>,
}

impl WhisperFilter {
    /// Returns the full parameters of the whisper filter.
    fn whisper_params(&self) -> FullParams {
        let mut params = FullParams::new(SamplingStrategy::default());

        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_non_speech_tokens(true);
        {
            let settings = self.settings.lock().unwrap();
            match settings.language.as_str() {
                "en" => params.set_language(Some("en")),
                "auto" => params.set_language(Some("auto")),
                other => panic!("unsupported language: {}", other),
            }
            params.set_translate(settings.translate);
            params.set_no_context(!settings.context);
        }
        params
    }
    /// Runs the whisper model on the given audio chunk and returns the resulting text segment as a buffer.
    fn run_model(&self, state: &mut State, chunk: Chunk) -> Result<Option<Buffer>, FlowError> {
        let samples = convert_integer_to_float_audio(&chunk.buffer);

        let start = Instant::now();
        state
            .whisper_state
            .full(self.whisper_params(), &samples)
            .unwrap();
        gstreamer::debug!(CAT, "model took {:?}", start.elapsed());

        let n_segments = state.whisper_state.full_n_segments().unwrap();
        if n_segments == 0 {
            return Ok(None);
        }
        let mut duration = 0; 
        let mut transcribed_text = String::new();
        for i in 0..n_segments {
            let segment = state.whisper_state.full_get_segment_text(i).ok().unwrap();
            let segment = segment
                .replace("[BLANK_AUDIO]", "")
                .replace("[ Silence ]", "")
                .replace("[silence]", "")
                .replace("(silence)", "")
                .replace("[ Pause ]", "")
                .trim()
                .to_owned();

            let start_ts = state.whisper_state.full_get_segment_t0(i).unwrap();
            let end_ts = state.whisper_state.full_get_segment_t1(i).unwrap();

            let segment = format!("{}\n", segment);
            transcribed_text.push_str(&segment);
            duration = duration + (end_ts as u64 - start_ts as u64) * 10;
        }
        let mut buffer = Buffer::with_size(transcribed_text.len()).map_err(|_| FlowError::Error)?;
        let buffer_mut = buffer.get_mut().ok_or(FlowError::Error)?;
        buffer_mut.set_pts(chunk.start_pts);
        buffer_mut.set_duration(ClockTime::from_mseconds(duration));
        buffer_mut
                .copy_from_slice(0, transcribed_text.as_bytes())
                .map_err(|_| FlowError::Error)?;
        
        gstreamer::info!(
            CAT,
            "Start pts: {:?}, duration: {:?}, text: {:?}",
            buffer.pts().unwrap(),
            buffer.duration().unwrap(),
            transcribed_text
        );
        Ok(Some(buffer))
    }
}

/// This is an implementation of the `ObjectSubclass` trait for the `WhisperFilter` struct.
/// It specifies the parent type as `BaseTransform` and the type as `super::WhisperFilter`.
/// The `NAME` constant is set to "GstWhisperFilter".
/// The `new` method creates a new instance of `WhisperFilter` with default settings.
/// The `settings` field is a `Mutex` that contains the filter's settings.
/// The `state` field is a `Mutex` that contains the filter's state.
#[glib::object_subclass]
impl ObjectSubclass for WhisperFilter {
    type ParentType = BaseTransform;
    type Type = super::WhisperFilter;

    const NAME: &'static str = "GstWhisperFilter";

    fn new() -> Self {
        Self {
            settings: Mutex::new(Settings {
                use_vad: DEFAULT_USE_VAD,
                vad_mode: DEFAULT_VAD_MODE.into(),
                min_voice_activity_ms: DEFAULT_MIN_VOICE_ACTIVITY_MS,
                language: DEFAULT_LANGUAGE.into(),
                translate: DEFAULT_TRANSLATE,
                context: DEFAULT_CONTEXT,
            }),
            state: Mutex::new(None),
        }
    }
}

/// Implementation of GObject's ObjectImpl trait for WhisperFilter.
impl ObjectImpl for WhisperFilter {
    /// Returns an array of ParamSpecs for the properties of the WhisperFilter.
    fn properties() -> &'static [ParamSpec] {
        static PROPERTIES: Lazy<Vec<ParamSpec>> = Lazy::new(|| {
            vec![
            glib::ParamSpecBoolean::builder("use-vad")
            .nick("Use VAD")
            .blurb(&format!("Whether to use VAD. Defaults to {}.", DEFAULT_USE_VAD))
            .mutable_ready()
            .mutable_paused()
            .mutable_playing()
            .build(),
            glib::ParamSpecString::builder("vad-mode")
            .nick("VAD mode")
            .blurb(&format!("The aggressiveness of voice detection. Defaults to '{}'. Other options are 'low-bitrate', 'aggressive' and 'very-aggressive'.", DEFAULT_VAD_MODE))
            .mutable_ready()
            .mutable_paused()
            .mutable_playing()
            .build(),
            glib::ParamSpecInt::builder("min-voice-activity-ms")
            .nick("Minimum voice activity")
            .blurb(&format!("The minimum duration of voice that must be detected for the model to run, in milliseconds. Defaults to {}ms.", DEFAULT_MIN_VOICE_ACTIVITY_MS))
            .mutable_ready()
            .mutable_paused()
            .mutable_playing()
            .build(),
            glib::ParamSpecString::builder("language")
            .nick("Language")
            .blurb(&format!("The target language. Defaults to '{}'. Specify 'auto' to use language detection.", DEFAULT_LANGUAGE))
            .mutable_ready()
            .mutable_paused()
            .mutable_playing()
            .build(),
            glib::ParamSpecBoolean::builder("translate")
            .nick("Translate")
            .blurb(&format!("Whether to translate into the target language. Defaults to {}.", DEFAULT_TRANSLATE))
            .mutable_ready()
            .mutable_paused()
            .mutable_playing()
            .build(),
            glib::ParamSpecBoolean::builder("context")
            .nick("Context")
            .blurb(&format!("Whether to use previous tokens as context for the model. Defaults to {}.", DEFAULT_CONTEXT))
            .mutable_ready()
            .mutable_paused()
            .mutable_playing()
            .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    /// Sets the value of a property of the WhisperFilter.
    fn set_property(&self, _id: usize, value: &Value, pspec: &ParamSpec) {
        let mut settings = self.settings.lock().unwrap();
        match pspec.name() {
            "use-vad" => {
                settings.use_vad = value.get().unwrap();
            }
            "vad-mode" => {
                settings.vad_mode = value.get().unwrap();
            }
            "min-voice-activity-ms" => {
                settings.min_voice_activity_ms = value.get().unwrap();
            }
            "language" => {
                settings.language = value.get().unwrap();
            }
            "translate" => {
                settings.translate = value.get().unwrap();
            }
            "context" => {
                settings.context = value.get().unwrap();
            }
            other => panic!("no such property: {}", other),
        }
    }

    /// Gets the value of a property of the WhisperFilter.
    fn property(&self, _id: usize, pspec: &ParamSpec) -> Value {
        let settings = self.settings.lock().unwrap();
        match pspec.name() {
            "use-vad" => settings.use_vad.to_value(),
            "vad-mode" => settings.vad_mode.to_value(),
            "min-voice-activity-ms" => settings.min_voice_activity_ms.to_value(),
            "language" => settings.language.to_value(),
            "translate" => settings.translate.to_value(),
            "context" => settings.context.to_value(),
            other => panic!("no such property: {}", other),
        }
    }
}

/// Implementation of the `GstObject` trait for the `WhisperFilter` struct.
impl GstObjectImpl for WhisperFilter {}

/// Implementation of the `ElementImpl` trait for the `WhisperFilter` struct.
impl ElementImpl for WhisperFilter {
    /// Returns the metadata of the element.
    fn metadata() -> Option<&'static ElementMetadata> {
        static ELEMENT_METADATA: Lazy<ElementMetadata> = Lazy::new(|| {
            ElementMetadata::new(
                "Transcriber",
                "Audio/Text/Filter",
                "Speech to text filter using Whisper",
                "Jasper Hugo <jasper@avstack.io>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    /// Returns the pad templates of the element.
    fn pad_templates() -> &'static [PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<PadTemplate>> = Lazy::new(|| {
            let src_pad_template =
                PadTemplate::new("src", PadDirection::Src, PadPresence::Always, &SRC_CAPS).unwrap();

            let sink_pad_template = gstreamer::PadTemplate::new(
                "sink",
                gstreamer::PadDirection::Sink,
                gstreamer::PadPresence::Always,
                &SINK_CAPS,
            )
            .unwrap();

            vec![src_pad_template, sink_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }
}

/// Implementation of the WhisperFilter struct.
impl WhisperFilter {
    /// Reads the audio samples from the given buffer and returns them as a vector of signed 16-bit integers.
    fn read_samples(&self, buffer: &Buffer) -> Result<Vec<i16>, FlowError> {
        let buffer_reader = buffer
            .as_ref()
            .map_readable()
            .map_err(|_| FlowError::Error)?;
        let samples = buffer_reader.as_slice_of().map_err(|_| FlowError::Error)?;
        gstreamer::debug!(CAT, "reading {} samples", samples.len());

        Ok(samples.to_vec())
    }
    /// Creates a new buffer for voice activity detection (VAD) based on the given samples.
    fn new_vad_buffer(&self, samples: &[i16]) -> Option<Vec<i16>> {
        let buffer_len = samples.len();
        if buffer_len >= 160 {
            let vad_buffer = if buffer_len >= 480 {
                &samples[0..480]
            } else if buffer_len >= 320 {
                &samples[0..320]
            } else {
                &samples[0..160]
            };
            Some(vad_buffer.to_vec())
        } else {
            None
        }
    }
    /// Handles voice activity by either extending the current chunk buffer or creating a new chunk buffer.
    fn handle_voice_activity(
        &self,
        state: &mut State,
        samples: &[i16],
        buffer: &Buffer,
    ) -> Result<GenerateOutputSuccess, FlowError> {
        if let Some(chunk) = state.chunk.as_mut() {
            gstreamer::debug!(CAT, "voice activity is on-going");
            chunk.buffer.extend_from_slice(samples);
        } else {
            gstreamer::debug!(CAT, "voice activity started");
            state.chunk = Some(Chunk {
                start_pts: buffer.pts().unwrap(),
                buffer: state
                    .prev_buffer
                    .drain(..)
                    .chain(samples.iter().copied())
                    .collect(),
            });
        }

        Ok(GenerateOutputSuccess::NoOutput)
    }
    /// Handles the end of voice activity by storing the previous buffer, checking if there is a chunk of audio to process, and running the model on the chunk if it exists.
    /// If the duration of the voice activity is less than the minimum voice activity duration, the function discards the voice activity and returns `GenerateOutputSuccess::NoOutput`.
    fn handle_voice_activity_boundary(
        &self,
        state: &mut State,
        samples: &[i16],
        buffer: &Buffer,
    ) -> Result<GenerateOutputSuccess, FlowError> {
        state.prev_buffer = samples.to_vec();

        if let Some(chunk) = state.chunk.take() {
            gstreamer::info!(CAT, "voice activity ended");
            let min_voice_activity_ms = self.settings.lock().unwrap().min_voice_activity_ms;
            if (buffer.pts().unwrap() - chunk.start_pts).mseconds() >= min_voice_activity_ms {
                let maybe_buffer = self.run_model(state, chunk)?;
                Ok(maybe_buffer
                    .map(GenerateOutputSuccess::Buffer)
                    .unwrap_or(GenerateOutputSuccess::NoOutput))
            } else {
                gstreamer::warning!(
                    CAT,
                    "discarding voice activity < {}ms",
                    min_voice_activity_ms
                );
                Ok(GenerateOutputSuccess::NoOutput)
            }
        } else {
            Ok(GenerateOutputSuccess::NoOutput)
        }
    }
    /// Handles end-of-stream event by running the model on the last audio chunk and pushing the output buffer to the source pad.
    fn handle_eos(&self, state: &mut State) {
        if let Some(chunk) = state.chunk.take() {
            gstreamer::info!(CAT, "voice activity ended");
            let maybe_buffer = self.run_model(state, chunk);
            let srcpad = self.obj().static_pad("src").unwrap();
            if let Ok(Some(buffer)) = maybe_buffer {
                let _ = srcpad.push(buffer);
            }
        }
    }
}

/// Implementation of the `BaseTransformImpl` trait for the `WhisperFilter` struct.    
impl BaseTransformImpl for WhisperFilter {
    const MODE: BaseTransformMode = BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    /// Starts the filter with the specified settings. 
    fn start(&self) -> Result<(), ErrorMessage> {
        gstreamer::debug!(CAT, "starting");
        let vad_mode = match self.settings.lock().unwrap().vad_mode.as_str() {
            "quality" => VadMode::Quality,
            "low-bitrate" => VadMode::LowBitrate,
            "aggressive" => VadMode::Aggressive,
            "very-aggressive" => VadMode::VeryAggressive,
            other => panic!("invalid VAD mode: {}", other),
        };

        let use_vad = self.settings.lock().unwrap().use_vad;

        if use_vad {
            *self.state.lock().unwrap() = Some(State {
                whisper_state: WHISPER_CONTEXT.create_state().unwrap(),
                voice_activity_detector: Some(vad::VoiceActivityDetector::new(vad_mode)),
                chunk: None,
                prev_buffer: Vec::new(),
            })
        } else {
            *self.state.lock().unwrap() = Some(State {
                whisper_state: WHISPER_CONTEXT.create_state().unwrap(),
                voice_activity_detector: None,
                chunk: None,
                prev_buffer: Vec::new(),
            })
        }
        gstreamer::debug!(CAT, "started");
        Ok(())
    }

    /// Stops the filter.
    fn stop(&self) -> Result<(), ErrorMessage> {
        gstreamer::debug!(CAT, "stopping");
        let _ = self.state.lock().unwrap().take();
        gstreamer::debug!(CAT, "stopped");
        Ok(())
    }

    /// Transforms the given caps based on the direction and an optional filter.
    fn transform_caps(
        &self,
        direction: PadDirection,
        _caps: &Caps,
        maybe_filter: Option<&Caps>,
    ) -> Option<Caps> {
        let mut caps = if direction == PadDirection::Src {
            SINK_CAPS.clone()
        } else {
            SRC_CAPS.clone()
        };
        if let Some(filter) = maybe_filter {
            caps = filter.intersect_with_mode(&caps, CapsIntersectMode::First);
        }
        Some(caps)
    }

    /// This function generates output from the filter. 
    /// It reads samples from the queued buffer and handles voice activity based on the settings. 
    /// If the `use_vad` setting is enabled, it uses the voice activity detector to determine if the buffer contains a voice segment. 
    /// If a voice segment is detected, it calls `handle_voice_activity` to handle the segment. 
    /// If not, it calls `handle_voice_activity_boundary` to handle the boundary of the voice activity. 
    /// If `use_vad` is disabled, it checks if the buffer is long enough to be considered a voice segment and calls the appropriate function. 
    /// If there are no queued buffers, it returns `GenerateOutputSuccess::NoOutput`.
    fn generate_output(&self) -> Result<GenerateOutputSuccess, FlowError> {
        if let Some(buffer) = self.take_queued_buffer() {
            let mut locked_state = self.state.lock().unwrap();
            let state = locked_state.as_mut().ok_or_else(|| {
                element_imp_error!(
                    self,
                    CoreError::Negotiation,
                    ["Can not generate an output without state"]
                );
                FlowError::NotNegotiated
            })?;
            let samples = self.read_samples(&buffer)?;
            let use_vad = self.settings.lock().unwrap().use_vad;
            if use_vad {
                let vad_buffer = self.new_vad_buffer(&samples);
                if let Some(vad_buffer) = vad_buffer {
                    if let Some(vad) = &state.voice_activity_detector {
                        if vad.is_voice_segment(&vad_buffer).unwrap() {
                            return self.handle_voice_activity(state, &samples, &buffer);
                        }
                    }
                }
                self.handle_voice_activity_boundary(state, &samples, &buffer)
            } else {
                if let Some(chunk) = state.chunk.as_mut() {
                    if (buffer.pts().unwrap() - chunk.start_pts).mseconds() >= DEFAULT_LENGTH_MS {
                        return self.handle_voice_activity_boundary(state, &samples, &buffer);
                    }
                }
                return self.handle_voice_activity(state, &samples, &buffer);
            }
        } else {
            gstreamer::debug!(CAT, "no queued buffers to take");
            Ok(GenerateOutputSuccess::NoOutput)
        }
    }

    /// Handle sink events, such as EOS (end-of-stream) events.
    fn sink_event(&self, event: gstreamer::Event) -> bool {
        if let EventView::Eos(_) = event.view() {
            gstreamer::info!(CAT, imp: self, "Handling EOS");
            let mut locked_state = self.state.lock().unwrap();
            let state = locked_state.as_mut().unwrap();
            self.handle_eos(state);
        }
        self.parent_sink_event(event)
    }
}
