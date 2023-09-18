use silero_vad_rs::model::{VadIterator, VadResult};

use std::{env, path::Path};

use super::SAMPLE_RATE;

/// Voice Activity Detector struct used for detecting voice activity in audio streams.
pub struct VoiceActivityDetector {
    vad: VadIterator,
}

impl VoiceActivityDetector {
    /// Creates a new instance of `Vad` with the specified `vad_mode`.
    pub fn new() -> Self {
        let path_str = env::var("SILERO_VAD_MODEL_PATH").unwrap();
        let path = Path::new(&path_str);

        let sampling_rate: i32 = SAMPLE_RATE as i32;
        let threshold = 0.4f32;
        let threshold_margin = 0.04f32;
        let min_silence_duration_ms = 60;
        let speech_pad_ms = 0;

        let vad = VadIterator::new(
            &path,
            sampling_rate,
            threshold,
            threshold_margin,
            min_silence_duration_ms,
            speech_pad_ms
        )
        .unwrap();

        Self { vad }
    }

    /// Determines whether the given buffer contains a voice segment or not using Voice Activity Detection (VAD).
    ///
    /// # Arguments
    ///
    /// * `buffer` - An array slice of 16-bit signed integers representing the audio buffer to be analyzed.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if the buffer contains a voice segment.
    /// * `Ok(false)` if the buffer does not contain a voice segment.
    /// * `Err(())` if there was an error sending the buffer to the VAD sender.
    pub fn is_voice_segment(&mut self, buffer: &[i16]) -> Result<bool, ()> {
        let buffer_f32: Vec<f32> = buffer.iter().map(|&x| x as f32).collect();
        self.vad.predict(&buffer_f32);
        let result = self.vad.result.as_ref().unwrap();
        let is_voice_segment = match result {
            VadResult::Start => true,
            VadResult::Speaking => true,
            VadResult::End => false,
            VadResult::Silence => false
        };
        Ok(is_voice_segment)
    }
}
