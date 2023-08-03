
use std::sync::atomic::Ordering;

use super::SAMPLE_RATE;

use webrtc_vad::Vad;

use std::thread;

use webrtc_vad::VadMode;

use std::sync::Condvar;

use std::sync::Mutex;

use std::sync::atomic::AtomicBool;

use std::sync::Arc;

use std::sync::mpsc;

pub(crate) struct VoiceActivityDetector {
    pub(crate) vad_sender: mpsc::Sender<Vec<i16>>,
    pub(crate) voice_activity_detected: Arc<AtomicBool>,
    pub(crate) result_ready: Arc<(Mutex<bool>, Condvar)>,
}

impl VoiceActivityDetector {
    pub fn new(vad_mode: VadMode) -> Self {
        let voice_activity_detected = Arc::new(AtomicBool::new(false));
        let result_ready = Arc::new((Mutex::new(false), Condvar::new()));
        let (vad_sender, vad_receiver) = mpsc::channel::<Vec<i16>>();
        let voice_activity_detected_clone = voice_activity_detected.clone();
        let result_ready_clone = result_ready.clone();

        thread::spawn(move || {
            let mut vad =
                Vad::new_with_rate_and_mode((SAMPLE_RATE as i32).try_into().unwrap(), vad_mode);
            while let Ok(next) = vad_receiver.recv() {
                let result = vad.is_voice_segment(&next).unwrap();

                let (lock, cvar) = &*result_ready_clone;
                let mut result_ready = lock.lock().unwrap();
                *result_ready = true;
                voice_activity_detected_clone.store(result, Ordering::Relaxed);
                cvar.notify_one();
            }
        });

        Self {
            vad_sender,
            voice_activity_detected,
            result_ready,
        }
    }

    pub fn is_voice_segment(&self, buffer: &[i16]) -> Result<bool, ()> {
        self.vad_sender.send(buffer.to_vec()).unwrap();

        let (lock, cvar) = &*self.result_ready;
        let mut result_ready = lock.lock().unwrap();
        while !*result_ready {
            result_ready = cvar.wait(result_ready).unwrap();
        }
        let result = self.voice_activity_detected.load(Ordering::Relaxed);
        *result_ready = false;
        self.voice_activity_detected.store(false, Ordering::Relaxed);

        Ok(result)
    }
}
