//! Portrait, skin & body retouching in scene-linear Rec.2020 (via perceptual Oklab):
//! - **Skin Retouching**: frequency-separation smoothing (`skin_smooth`, `skin_texture`), chroma
//!   tone evening (`skin_tone_even`), specular shine compression (`shine_control`), under-eye
//!   dark-circle & eye-bag reduction (`dark_circles`), and wrinkle/line attenuation (`wrinkle_reduce`).
//! - **Skin Tone & Radiance**: subsurface radiance/brilliance (`skin_radiance`), golden warmth/tan
//!   (`skin_warmth`), and facial redness/rosacea control (`skin_redness`).
//! - **Eyes & Brows**: iris brightening (`eye_brighten`), sclera whitening & red-vein removal
//!   (`eye_whiten`), iris/lash micro-contrast (`eye_clarity`), and eyebrow definition (`eyebrow_enhance`).
//! - **Mouth, Teeth & Lips**: teeth whitening & yellow-cast removal (`teeth_whiten`), lip tint &
//!   richness (`lip_enhance`), and lip gloss/moisture fullness (`lip_moisture`).
//! - **Face Sculpt & 2D Reshaping**: 3D dodge & burn facial contouring (`face_contour`) plus
//!   C2-continuous inverse-mapped bilinear mesh warps (`face_slim`, `eye_enlarge`, `nose_slim`,
//!   `chin_length`, `smile_lift`).
//! - **Hair & Flyaways**: hair shineband luster (`hair_luster`) and stray-hair/flyaway attenuation
//!   (`stray_hair`).

use lightcraft_color::perceptual::{oklab_from_2020, oklab_to_2020};
use lightcraft_develop::{DevelopSettings, Retouch};
use lightcraft_raster::blur::gaussian;
use lightcraft_raster::{Plane, Rgb32f, par_join};

use crate::for_rows;
use crate::local::guided_fast;

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-6 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// C2-continuous bell kernel `(1 - r^2)^2` for `r = hypot(dx, dy) < 1`.
#[inline]
fn bell2(dx: f32, dy: f32) -> f32 {
    let r2 = dx * dx + dy * dy;
    if r2 >= 1.0 {
        0.0
    } else {
        let t = 1.0 - r2;
        t * t
    }
}

/// Smooth perceptual skin-locus weight in Oklab (`0.0..=1.0`) covering Fitzpatrick types I–VI
/// across studio, daylight, and indoor lighting in Rec.2020.
#[inline]
pub fn skin_weight(lab: [f32; 3]) -> f32 {
    let [l, a, b] = lab;
    if !l.is_finite() || !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    // Skin tones are warm (a > 0 and b > 0) or rosy/fair (a > 0, b slightly negative) with
    // low-to-moderate chroma and non-extreme lightness. Rec.2020 primaries produce slightly smaller
    // (a, b) chroma than sRGB, so keep the lower chroma/lightness ramps generous while still
    // rejecting negative-a sky and foliage.
    let c = a.hypot(b);
    let hue = b.atan2(a); // radians; human skin clusters around -0.25..1.48 rad (~-14°..85°)
    let w_l = smoothstep(0.05, 0.16, l) * (1.0 - smoothstep(0.92, 0.995, l));
    let w_c = smoothstep(0.003, 0.013, c) * (1.0 - smoothstep(0.24, 0.36, c));
    let w_h = smoothstep(-0.32, 0.06, hue) * (1.0 - smoothstep(1.30, 1.56, hue));
    (w_l * w_c * w_h).clamp(0.0, 1.0)
}

/// Perceptual lip-locus weight in Oklab (`0.0..=1.0`): stronger rose/magenta-red (`a`) relative to
/// yellow (`b`) than surrounding skin.
#[inline]
pub fn lip_weight(lab: [f32; 3]) -> f32 {
    let [l, a, b] = lab;
    if !l.is_finite() || !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    let w_l = smoothstep(0.12, 0.24, l) * (1.0 - smoothstep(0.82, 0.92, l));
    let w_a = smoothstep(0.016, 0.044, a);
    let w_rose = smoothstep(-0.020, 0.012, a - b * 0.65);
    (w_l * w_a * w_rose).clamp(0.0, 1.0)
}

/// Perceptual teeth-locus weight in Oklab (`0.0..=1.0`): bright, low-to-warm ivory chroma with
/// local contrast against surrounding lips/mouth.
#[inline]
pub fn teeth_weight(lab: [f32; 3], detail: f32) -> f32 {
    let [l, a, b] = lab;
    if !l.is_finite() || !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    let c = a.hypot(b);
    let w_l = smoothstep(0.32, 0.50, l) * (1.0 - smoothstep(0.97, 0.999, l));
    let w_c = 1.0 - smoothstep(0.065, 0.145, c);
    let w_a = 1.0 - smoothstep(0.028, 0.062, a.abs());
    let w_b = smoothstep(-0.030, -0.004, b) * (1.0 - smoothstep(0.115, 0.180, b));
    let w_edge = 0.55 + 0.45 * smoothstep(0.002, 0.024, detail);
    (w_l * w_c * w_a * w_b * w_edge).clamp(0.0, 1.0)
}

/// Perceptual sclera (eye-white) weight in Oklab (`0.0..=1.0`): bright, low-chroma non-skin pixels.
#[inline]
pub fn sclera_weight(lab: [f32; 3], w_skin: f32) -> f32 {
    let [l, a, b] = lab;
    if !l.is_finite() || !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    let c = a.hypot(b);
    let w_l = smoothstep(0.24, 0.42, l) * (1.0 - smoothstep(0.97, 0.998, l));
    let w_c = 1.0 - smoothstep(0.045, 0.095, c);
    (w_l * w_c * (1.0 - 0.65 * w_skin)).clamp(0.0, 1.0)
}

/// Quantitative portrait & skin telemetry extracted by YuNet ONNX + Oklab frequency separation
/// for the AI Retouch Director (Gemini / OpenAI endpoint) and smart auto-retouching.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortraitAnalysis {
    pub faces_detected: usize,
    pub neural_detector: bool,
    pub primary_face_box: [f32; 4],
    pub landmarks: [[f32; 2]; 5],
    pub skin_coverage: f32,
    pub skin_lightness: f32,
    pub skin_redness: f32,
    pub skin_warmth: f32,
    pub skin_unevenness: f32,
    pub shine_level: f32,
    pub dark_circle_depth: f32,
    pub wrinkle_energy: f32,
    pub eye_whiteness: f32,
    pub teeth_yellowness: f32,
    pub lip_saturation: f32,
    pub scene_exposure_ev: f32,
    pub recommended: Retouch,
}

/// Fast cached neural face detection on a scene-linear Rec.2020 image using embedded YuNet 2023mar.
fn detect_neural_faces_cached(img: &Rgb32f) -> Vec<lightcraft_faces::yunet::Face> {
    use std::sync::{Mutex, PoisonError};
    static CACHE: Mutex<Vec<(u64, Vec<lightcraft_faces::yunet::Face>)>> = Mutex::new(Vec::new());

    let (w, h) = (img.width, img.height);
    if w < 48 || h < 48 || img.data.len() < w * h {
        return Vec::new();
    }
    let Some(detector) = lightcraft_faces::yunet::builtin() else {
        return Vec::new();
    };

    // Compute a fast 64-point fingerprint so interactive slider drags and multi-resolution renders hit the cache in O(1).
    let mut hash: u64 = ((w as u64) << 32) ^ (h as u64) ^ 0x9e37_79b9_7f4a_7c15;
    let step = (img.data.len() / 64).max(1);
    for px in img.data.iter().step_by(step).take(64) {
        let r = px[0].to_bits() as u64;
        let g = px[1].to_bits() as u64;
        let b = px[2].to_bits() as u64;
        hash = hash.wrapping_mul(0x517c_c1b7_2722_0a95) ^ r ^ g.rotate_left(21) ^ b.rotate_left(42);
    }

    if let Some((_, faces)) = CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .find(|(h_key, _)| *h_key == hash)
    {
        return faces.clone();
    }

    // Downsample to <= 288 px long edge and auto-normalize exposure to 8-bit sRGB for YuNet inference (~2 ms).
    let max_edge = 288usize;
    let scale = (max_edge as f32 / w.max(h) as f32).min(1.0);
    let dw = ((w as f32 * scale).round() as usize).max(32);
    let dh = ((h as f32 * scale).round() as usize).max(32);
    let mut sum_lum = 0.0f64;
    let total_px = (dw * dh).max(1);
    for y in (0..dh).step_by(2) {
        let sy = ((y * h) / dh).min(h - 1);
        let row = sy * w;
        for x in (0..dw).step_by(2) {
            let sx = ((x * w) / dw).min(w - 1);
            let px = img.data.get(row + sx).copied().unwrap_or([0.0; 3]);
            let lum = 0.2627 * px[0].max(0.0) + 0.6780 * px[1].max(0.0) + 0.0593 * px[2].max(0.0);
            if lum.is_finite() {
                sum_lum += f64::from(lum.min(2.0));
            }
        }
    }
    let mean_lum = (sum_lum / ((total_px / 4).max(1) as f64)) as f32;
    let norm_gain = if mean_lum > 1e-4 && mean_lum < 0.14 {
        (0.18 / mean_lum).clamp(1.0, 8.0)
    } else {
        1.0
    };

    let mut rgb = Vec::with_capacity(total_px * 3);
    for y in 0..dh {
        let sy = ((y * h) / dh).min(h - 1);
        let row = sy * w;
        for x in 0..dw {
            let sx = ((x * w) / dw).min(w - 1);
            let px = img.data.get(row + sx).copied().unwrap_or([0.0; 3]);
            // Convert linear Rec.2020 through gamma ~2.2 to approximate display sRGB 0..255
            let r = ((px[0].max(0.0) * norm_gain).clamp(0.0, 1.0).powf(0.4545) * 255.0).round() as u8;
            let g = ((px[1].max(0.0) * norm_gain).clamp(0.0, 1.0).powf(0.4545) * 255.0).round() as u8;
            let b = ((px[2].max(0.0) * norm_gain).clamp(0.0, 1.0).powf(0.4545) * 255.0).round() as u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }

    let opts = lightcraft_faces::yunet::Options { score: 0.38, nms_iou: 0.3, max_faces: 12 };
    let faces = detector.detect(&rgb, dw, dh, &opts).unwrap_or_default();
    let mut guard = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    guard.retain(|(k, _)| *k != hash);
    if guard.len() >= 16 {
        guard.remove(0);
    }
    guard.push((hash, faces.clone()));
    faces
}

/// Detected or canonical portrait face geometry in pixel coordinates, backed by YuNet 5-point
/// neural landmarks when a face is present and Oklab skin statistical moments as fallback.
#[derive(Clone, Copy, Debug)]
struct FaceZone {
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    right_eye: (f32, f32),
    left_eye: (f32, f32),
    nose_tip: (f32, f32),
    mouth_right: (f32, f32),
    mouth_left: (f32, f32),
    mouth_y: f32,
    neural: bool,
}

impl FaceZone {
    /// Locate the primary face and 5 anatomical landmarks via embedded YuNet ONNX neural inference,
    /// falling back smoothly to Oklab skin/lip moments when no neural face is detected.
    fn detect(img: &Rgb32f, w: usize, h: usize, skin: &Plane, a_plane: &Plane, b_plane: &Plane, l_plane: &Plane) -> Self {
        let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);

        // 1. Try YuNet ONNX neural face + 5-landmark detector first.
        let neural_faces = detect_neural_faces_cached(img);
        if let Some(best) = neural_faces.iter().max_by(|a, b| {
            let area_a = (a.x1 - a.x0) * (a.y1 - a.y0) * a.score;
            let area_b = (b.x1 - b.x0) * (b.y1 - b.y0) * b.score;
            area_a.total_cmp(&area_b)
        }) {
            let cx = ((best.x0 + best.x1) * 0.5 * wf).clamp(0.0, wf);
            let cy = ((best.y0 + best.y1) * 0.5 * hf).clamp(0.0, hf);
            let rx = ((best.x1 - best.x0) * 0.54 * wf).max(8.0);
            let ry = ((best.y1 - best.y0) * 0.56 * hf).max(8.0);
            let right_eye = (best.landmarks[0].0.clamp(0.0, 1.0) * wf, best.landmarks[0].1.clamp(0.0, 1.0) * hf);
            let left_eye = (best.landmarks[1].0.clamp(0.0, 1.0) * wf, best.landmarks[1].1.clamp(0.0, 1.0) * hf);
            let nose_tip = (best.landmarks[2].0.clamp(0.0, 1.0) * wf, best.landmarks[2].1.clamp(0.0, 1.0) * hf);
            let mouth_right = (best.landmarks[3].0.clamp(0.0, 1.0) * wf, best.landmarks[3].1.clamp(0.0, 1.0) * hf);
            let mouth_left = (best.landmarks[4].0.clamp(0.0, 1.0) * wf, best.landmarks[4].1.clamp(0.0, 1.0) * hf);
            let mouth_y = (mouth_right.1 + mouth_left.1) * 0.5;
            return Self {
                cx,
                cy,
                rx,
                ry,
                right_eye,
                left_eye,
                nose_tip,
                mouth_right,
                mouth_left,
                mouth_y,
                neural: true,
            };
        }

        // 2. Fallback: Oklab skin/lip statistical moments (for close-up skin crops or non-frontal portraits).
        let (mut sum_w, mut sum_x, mut sum_y) = (0.0f64, 0.0f64, 0.0f64);
        let step = (w.min(h) / 64).max(1);
        for y in (0..h).step_by(step) {
            let py = y as f32 + 0.5;
            let vy = 1.0 - 0.45 * smoothstep(0.60 * hf, 0.92 * hf, py);
            let row = y * w;
            for x in (0..w).step_by(step) {
                let s = skin.data.get(row + x).copied().unwrap_or(0.0) * vy;
                if s > 0.15 {
                    let sw = f64::from(s);
                    sum_w += sw;
                    sum_x += sw * f64::from(x as f32 + 0.5);
                    sum_y += sw * f64::from(py);
                }
            }
        }

        let total_samples = ((w / step).max(1) * (h / step).max(1)) as f64;
        if sum_w < total_samples * 0.015 {
            let (cx, cy, rx, ry) = (wf * 0.5, hf * 0.44, (wf * 0.26).max(8.0), (hf * 0.32).max(8.0));
            let eye_y = cy - ry * 0.16;
            let eye_dx = rx * 0.38;
            let mouth_y = cy + ry * 0.44;
            return Self {
                cx,
                cy,
                rx,
                ry,
                right_eye: (cx - eye_dx, eye_y),
                left_eye: (cx + eye_dx, eye_y),
                nose_tip: (cx, cy + ry * 0.08),
                mouth_right: (cx - rx * 0.28, mouth_y),
                mouth_left: (cx + rx * 0.28, mouth_y),
                mouth_y,
                neural: false,
            };
        }

        let cx = (sum_x / sum_w) as f32;
        let cy = (sum_y / sum_w) as f32;
        let (mut var_x, mut var_y) = (0.0f64, 0.0f64);
        for y in (0..h).step_by(step) {
            let py = y as f32 + 0.5;
            let vy = 1.0 - 0.45 * smoothstep(0.60 * hf, 0.92 * hf, py);
            let row = y * w;
            for x in (0..w).step_by(step) {
                let s = skin.data.get(row + x).copied().unwrap_or(0.0) * vy;
                if s > 0.15 {
                    let sw = f64::from(s);
                    let dx = f64::from(x as f32 + 0.5 - cx);
                    let dy = f64::from(py - cy);
                    var_x += sw * dx * dx;
                    var_y += sw * dy * dy;
                }
            }
        }
        let std_x = (var_x / sum_w).sqrt() as f32;
        let std_y = (var_y / sum_w).sqrt() as f32;
        let rx = (std_x * 1.85).clamp(wf * 0.12, wf * 0.45).max(8.0);
        let ry = (std_y * 2.05).clamp(hf * 0.14, hf * 0.48).max(8.0);

        let mut lip_w = 0.0f64;
        let mut lip_y = 0.0f64;
        let y0 = ((cy + ry * 0.12).max(0.0) as usize).min(h);
        let y1 = ((cy + ry * 0.78).max(0.0) as usize).min(h);
        let x0 = ((cx - rx * 0.50).max(0.0) as usize).min(w);
        let x1 = ((cx + rx * 0.50).max(0.0) as usize).min(w);
        for y in (y0..y1).step_by(step) {
            let row = y * w;
            for x in (x0..x1).step_by(step) {
                let i = row + x;
                let lab = [
                    l_plane.data.get(i).copied().unwrap_or(0.0),
                    a_plane.data.get(i).copied().unwrap_or(0.0),
                    b_plane.data.get(i).copied().unwrap_or(0.0),
                ];
                let lw = lip_weight(lab);
                if lw > 0.25 {
                    lip_w += f64::from(lw);
                    lip_y += f64::from(lw) * f64::from(y as f32 + 0.5);
                }
            }
        }
        let mouth_y = if lip_w > 2.0 {
            ((lip_y / lip_w) as f32).clamp(cy + ry * 0.22, cy + ry * 0.68)
        } else {
            cy + ry * 0.44
        };
        let eye_y = cy - ry * 0.16;
        let eye_dx = rx * 0.38;

        Self {
            cx,
            cy,
            rx,
            ry,
            right_eye: (cx - eye_dx, eye_y),
            left_eye: (cx + eye_dx, eye_y),
            nose_tip: (cx, cy + ry * 0.08),
            mouth_right: (cx - rx * 0.28, mouth_y),
            mouth_left: (cx + rx * 0.28, mouth_y),
            mouth_y,
            neural: false,
        }
    }
}

/// Default recommended portrait retouch recipe used as a baseline when image telemetry is unavailable.
fn default_recommended_retouch() -> Retouch {
    Retouch {
        skin_smooth: 58.0,
        skin_texture: 44.0,
        skin_tone_even: 50.0,
        shine_control: 40.0,
        dark_circles: 45.0,
        wrinkle_reduce: 30.0,
        skin_radiance: 22.0,
        skin_warmth: 8.0,
        skin_redness: -12.0,
        eye_brighten: 38.0,
        eye_whiten: 30.0,
        eye_clarity: 32.0,
        eyebrow_enhance: 16.0,
        teeth_whiten: 38.0,
        lip_enhance: 20.0,
        lip_moisture: 24.0,
        face_contour: 32.0,
        face_slim: 12.0,
        eye_enlarge: 10.0,
        nose_slim: 8.0,
        chin_length: 0.0,
        smile_lift: 10.0,
        body_slim: 0.0,
        waist_slim: 0.0,
        stomach_flat: 0.0,
        hip_enhance: 0.0,
        shoulder_width: 0.0,
        arm_slim: 0.0,
        neck_slim: 0.0,
        chest_enhance: 0.0,
        leg_length: 0.0,
        hair_luster: 25.0,
        stray_hair: 28.0,
        blemish_sensitivity: 55.0,
    }
}

/// Analyze a scene-linear Rec.2020 portrait image using YuNet ONNX neural landmarks and Oklab
/// frequency separation, returning quantitative telemetry and a recommended `Retouch` recipe.
pub fn analyze_portrait(img: &Rgb32f) -> PortraitAnalysis {
    let (w, h) = (img.width, img.height);
    let n = w * h;
    if w == 0 || h == 0 || img.data.len() < n {
        return PortraitAnalysis {
            faces_detected: 1,
            neural_detector: false,
            primary_face_box: [0.25, 0.15, 0.75, 0.85],
            landmarks: [
                [0.39, 0.38],
                [0.61, 0.38],
                [0.50, 0.50],
                [0.41, 0.63],
                [0.59, 0.63],
            ],
            skin_coverage: 0.25,
            skin_lightness: 0.65,
            skin_redness: 0.04,
            skin_warmth: 0.05,
            skin_unevenness: 38.0,
            shine_level: 28.0,
            dark_circle_depth: 32.0,
            wrinkle_energy: 22.0,
            eye_whiteness: 70.0,
            teeth_yellowness: 25.0,
            lip_saturation: 42.0,
            scene_exposure_ev: 0.0,
            recommended: default_recommended_retouch(),
        };
    }

    let mut l_plane = Plane::new(w, h);
    let mut a_plane = Plane::new(w, h);
    let mut b_plane = Plane::new(w, h);
    let mut mask_plane = Plane::new(w, h);
    let mut sum_l_all = 0.0f64;
    for i in 0..n {
        if let Some(&rgb) = img.data.get(i) {
            let lab = oklab_from_2020([rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]);
            if lab[0].is_finite() {
                sum_l_all += f64::from(lab[0]);
            }
            if let Some(v) = l_plane.data.get_mut(i) {
                *v = lab[0];
            }
            if let Some(v) = a_plane.data.get_mut(i) {
                *v = lab[1];
            }
            if let Some(v) = b_plane.data.get_mut(i) {
                *v = lab[2];
            }
        }
    }
    let mean_l = (sum_l_all / (n.max(1) as f64)) as f32;
    let cbrt_gain = if mean_l > 1e-3 && mean_l < 0.45 {
        (0.52 / mean_l).clamp(1.0, 2.2)
    } else {
        1.0
    };
    for i in 0..n {
        let lab = [
            l_plane.data.get(i).copied().unwrap_or(0.0) * cbrt_gain,
            a_plane.data.get(i).copied().unwrap_or(0.0) * cbrt_gain,
            b_plane.data.get(i).copied().unwrap_or(0.0) * cbrt_gain,
        ];
        if let Some(v) = mask_plane.data.get_mut(i) {
            *v = skin_weight(lab);
        }
    }

    let long = w.max(h) as f32;
    let skin_mask = gaussian(&mask_plane, (long * 0.003).clamp(1.0, 6.0));
    let neural_faces = detect_neural_faces_cached(img);
    let fz = FaceZone::detect(img, w, h, &skin_mask, &a_plane, &b_plane, &l_plane);

    let fine_sigma = (long * 0.0008).clamp(0.6, 2.2);
    let mid_sigma = (long * 0.020).clamp(3.0, 42.0);
    let low_in = gaussian(&l_plane, fine_sigma);
    let base_l = guided_fast(&low_in, mid_sigma, 0.022);

    let (wf, hf) = (w as f32, h as f32);
    let step = (w.min(h) / 96).max(1);
    let mut total_samples = 0.0f64;
    let mut skin_w_sum = 0.0f64;
    let mut sum_l = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_mid_abs = 0.0f64;
    let mut sum_crease = 0.0f64;
    let mut sum_shine = 0.0f64;
    let mut sum_dc = 0.0f64;
    let mut dc_w_sum = 0.0f64;
    let mut sclera_sum = 0.0f64;
    let mut sclera_w_sum = 0.0f64;
    let mut teeth_b_sum = 0.0f64;
    let mut teeth_w_sum = 0.0f64;
    let mut lip_c_sum = 0.0f64;
    let mut lip_w_sum = 0.0f64;

    for y in (0..h).step_by(step) {
        let py = y as f32 + 0.5;
        let v_face = (py - fz.cy) / fz.ry;
        let row = y * w;
        for x in (0..w).step_by(step) {
            total_samples += 1.0;
            let px_x = x as f32 + 0.5;
            let u_face = (px_x - fz.cx) / fz.rx;
            let in_face = bell2(u_face / 1.25, v_face / 1.35);
            if in_face <= 1e-3 {
                continue;
            }
            let i = row + x;
            let l0 = l_plane.data.get(i).copied().unwrap_or(0.0);
            let low = low_in.data.get(i).copied().unwrap_or(l0);
            let base = base_l.data.get(i).copied().unwrap_or(low);
            let a0 = a_plane.data.get(i).copied().unwrap_or(0.0);
            let b0 = b_plane.data.get(i).copied().unwrap_or(0.0);
            let high = l0 - low;
            let mid = low - base;
            let detail = high.abs() + 0.5 * mid.abs();
            let lab_eval = [l0 * cbrt_gain, a0 * cbrt_gain, b0 * cbrt_gain];
            let face_floor = 0.55 * bell2(u_face / 0.90, (v_face - 0.04) / 1.0) * smoothstep(0.12, 0.25, lab_eval[0]);
            let ws = ((skin_mask.data.get(i).copied().unwrap_or(0.0) * in_face).max(face_floor)).clamp(0.0, 1.0);

            if ws > 0.15 {
                let sw = f64::from(ws);
                skin_w_sum += sw;
                sum_l += sw * f64::from(l0);
                sum_a += sw * f64::from(a0);
                sum_b += sw * f64::from(b0);
                sum_mid_abs += sw * f64::from(mid.abs() + 0.5 * high.abs());
                if mid < -0.002 {
                    sum_crease += sw * f64::from(-mid);
                }
                if lab_eval[0] > 0.58 && l0 > base {
                    sum_shine += sw * f64::from((l0 - base) * smoothstep(0.55, 0.78, lab_eval[0]));
                }

                // Under-eye zones right below each eye landmark
                let eye_rx = (fz.rx * 0.34).max(4.0);
                let eye_ry = (fz.ry * 0.26).max(4.0);
                let dc_r = bell2((px_x - fz.right_eye.0) / eye_rx, (py - (fz.right_eye.1 + eye_ry * 0.72)) / eye_ry);
                let dc_l = bell2((px_x - fz.left_eye.0) / eye_rx, (py - (fz.left_eye.1 + eye_ry * 0.72)) / eye_ry);
                let dc_zone = dc_r.max(dc_l);
                if dc_zone > 0.1 {
                    let dw = sw * f64::from(dc_zone);
                    dc_w_sum += dw;
                    sum_dc += dw * f64::from((base - low).max(0.0));
                }
            }

            let sw_sclera = sclera_weight(lab_eval, ws) * in_face;
            if sw_sclera > 0.15 {
                sclera_w_sum += f64::from(sw_sclera);
                sclera_sum += f64::from(sw_sclera * lab_eval[0]);
            }

            let mouth_v = (py - fz.mouth_y) / (fz.ry * 0.32).max(4.0);
            let mouth_box = bell2(u_face / 0.55, mouth_v);
            if mouth_box > 0.1 {
                let wt = teeth_weight(lab_eval, detail) * mouth_box;
                if wt > 0.15 {
                    teeth_w_sum += f64::from(wt);
                    teeth_b_sum += f64::from(wt * lab_eval[2].max(0.0));
                }
                let wl = lip_weight(lab_eval) * mouth_box;
                if wl > 0.15 {
                    lip_w_sum += f64::from(wl);
                    lip_c_sum += f64::from(wl * lab_eval[1].hypot(lab_eval[2]));
                }
            }
        }
    }

    let skin_coverage = (skin_w_sum / total_samples.max(1.0)).clamp(0.0, 1.0) as f32;
    let skin_lightness = if skin_w_sum > 1e-3 { (sum_l / skin_w_sum) as f32 } else { 0.66 };
    let skin_redness = if skin_w_sum > 1e-3 { (sum_a / skin_w_sum) as f32 } else { 0.042 };
    let skin_warmth = if skin_w_sum > 1e-3 { (sum_b / skin_w_sum) as f32 } else { 0.052 };
    let skin_unevenness = if skin_w_sum > 1e-3 {
        (((sum_mid_abs / skin_w_sum) as f32 / 0.022) * 100.0).clamp(15.0, 95.0)
    } else {
        40.0
    };
    let shine_level = if skin_w_sum > 1e-3 {
        (((sum_shine / skin_w_sum) as f32 / 0.008) * 100.0).clamp(10.0, 95.0)
    } else {
        25.0
    };
    let dark_circle_depth = if dc_w_sum > 1e-3 {
        (((sum_dc / dc_w_sum) as f32 / 0.020) * 100.0).clamp(15.0, 95.0)
    } else {
        30.0
    };
    let wrinkle_energy = if skin_w_sum > 1e-3 {
        (((sum_crease / skin_w_sum) as f32 / 0.012) * 100.0).clamp(10.0, 95.0)
    } else {
        25.0
    };
    let eye_whiteness = if sclera_w_sum > 1e-3 {
        (((sclera_sum / sclera_w_sum) as f32) * 100.0).clamp(35.0, 95.0)
    } else {
        70.0
    };
    let teeth_yellowness = if teeth_w_sum > 1e-3 {
        (((teeth_b_sum / teeth_w_sum) as f32 / 0.040) * 100.0).clamp(10.0, 95.0)
    } else {
        25.0
    };
    let lip_saturation = if lip_w_sum > 1e-3 {
        (((lip_c_sum / lip_w_sum) as f32 / 0.14) * 100.0).clamp(15.0, 95.0)
    } else {
        45.0
    };
    let scene_exposure_ev = ((0.67 - skin_lightness) * 2.2).clamp(-1.5, 1.5);

    // Derive strong, visibly effective Retouch sliders from the quantitative telemetry:
    let rec_smooth = (38.0 + skin_unevenness * 0.52).clamp(42.0, 82.0).round() as f64;
    let rec_tone_even = (32.0 + skin_unevenness * 0.48).clamp(35.0, 78.0).round() as f64;
    let rec_shine = (24.0 + shine_level * 0.65).clamp(25.0, 82.0).round() as f64;
    let rec_dc = (28.0 + dark_circle_depth * 0.58).clamp(30.0, 80.0).round() as f64;
    let rec_wrinkle = (18.0 + wrinkle_energy * 0.52).clamp(20.0, 70.0).round() as f64;
    let rec_radiance = ((0.70 - skin_lightness).max(0.0) * 150.0 + 18.0).clamp(16.0, 48.0).round() as f64;
    let rec_redness = if skin_redness > 0.046 {
        (-((skin_redness - 0.040) * 900.0)).clamp(-38.0, -10.0).round() as f64
    } else {
        -8.0
    };
    let rec_warmth = if skin_warmth < 0.050 {
        ((0.054 - skin_warmth) * 650.0).clamp(6.0, 26.0).round() as f64
    } else {
        8.0
    };

    let recommended = Retouch {
        skin_smooth: rec_smooth,
        skin_texture: 44.0,
        skin_tone_even: rec_tone_even,
        shine_control: rec_shine,
        dark_circles: rec_dc,
        wrinkle_reduce: rec_wrinkle,
        skin_radiance: rec_radiance,
        skin_warmth: rec_warmth,
        skin_redness: rec_redness,
        eye_brighten: 38.0,
        eye_whiten: (100.0 - eye_whiteness as f64).clamp(20.0, 45.0).round(),
        eye_clarity: 34.0,
        eyebrow_enhance: 18.0,
        teeth_whiten: (22.0 + teeth_yellowness as f64 * 0.48).clamp(25.0, 60.0).round(),
        lip_enhance: 22.0,
        lip_moisture: 25.0,
        face_contour: 34.0,
        face_slim: 14.0,
        eye_enlarge: 10.0,
        nose_slim: 10.0,
        chin_length: 0.0,
        smile_lift: 12.0,
        body_slim: 0.0,
        waist_slim: 0.0,
        stomach_flat: 0.0,
        hip_enhance: 0.0,
        shoulder_width: 0.0,
        arm_slim: 0.0,
        neck_slim: 0.0,
        chest_enhance: 0.0,
        leg_length: 0.0,
        hair_luster: 26.0,
        stray_hair: 30.0,
        blemish_sensitivity: 55.0,
    };

    let primary_face_box = [
        ((fz.cx - fz.rx) / wf).clamp(0.0, 1.0),
        ((fz.cy - fz.ry) / hf).clamp(0.0, 1.0),
        ((fz.cx + fz.rx) / wf).clamp(0.0, 1.0),
        ((fz.cy + fz.ry) / hf).clamp(0.0, 1.0),
    ];
    let landmarks = [
        [fz.right_eye.0 / wf, fz.right_eye.1 / hf],
        [fz.left_eye.0 / wf, fz.left_eye.1 / hf],
        [fz.nose_tip.0 / wf, fz.nose_tip.1 / hf],
        [fz.mouth_right.0 / wf, fz.mouth_right.1 / hf],
        [fz.mouth_left.0 / wf, fz.mouth_left.1 / hf],
    ];

    PortraitAnalysis {
        faces_detected: if fz.neural { neural_faces.len() } else { usize::from(skin_coverage > 0.03) },
        neural_detector: fz.neural,
        primary_face_box,
        landmarks,
        skin_coverage,
        skin_lightness,
        skin_redness,
        skin_warmth,
        skin_unevenness,
        shine_level,
        dark_circle_depth,
        wrinkle_energy,
        eye_whiteness,
        teeth_yellowness,
        lip_saturation,
        scene_exposure_ev,
        recommended,
    }
}

#[inline]
fn clean_u01(v: f64) -> f32 {
    if v.is_finite() { (v / 100.0).clamp(0.0, 1.0) as f32 } else { 0.0 }
}

#[inline]
fn clean_s11(v: f64) -> f32 {
    if v.is_finite() { (v / 100.0).clamp(-1.0, 1.0) as f32 } else { 0.0 }
}

/// Apply 2D C2-continuous inverse bilinear mesh warp for facial & full-body reshaping
/// (`face_slim`, `eye_enlarge`, `nose_slim`, `chin_length`, `smile_lift`, `body_slim`,
/// `waist_slim`, `stomach_flat`, `hip_enhance`, `shoulder_width`, `arm_slim`, `neck_slim`,
/// `chest_enhance`, `leg_length`).
fn apply_reshape_warp(img: &mut Rgb32f, r: &Retouch, fz: &FaceZone) {
    let face_slim = clean_s11(r.face_slim);
    let eye_enlarge = clean_s11(r.eye_enlarge);
    let nose_slim = clean_s11(r.nose_slim);
    let chin_length = clean_s11(r.chin_length);
    let smile_lift = clean_s11(r.smile_lift);

    let body_slim = clean_s11(r.body_slim);
    let waist_slim = clean_s11(r.waist_slim);
    let stomach_flat = clean_u01(r.stomach_flat);
    let hip_enhance = clean_s11(r.hip_enhance);
    let shoulder_width = clean_s11(r.shoulder_width);
    let arm_slim = clean_s11(r.arm_slim);
    let neck_slim = clean_s11(r.neck_slim);
    let chest_enhance = clean_s11(r.chest_enhance);
    let leg_length = clean_s11(r.leg_length);

    let has_face_warp = face_slim.abs() > 1e-4
        || eye_enlarge.abs() > 1e-4
        || nose_slim.abs() > 1e-4
        || chin_length.abs() > 1e-4
        || smile_lift.abs() > 1e-4;
    let has_body_warp = body_slim.abs() > 1e-4
        || waist_slim.abs() > 1e-4
        || stomach_flat > 1e-4
        || hip_enhance.abs() > 1e-4
        || shoulder_width.abs() > 1e-4
        || arm_slim.abs() > 1e-4
        || neck_slim.abs() > 1e-4
        || chest_enhance.abs() > 1e-4
        || leg_length.abs() > 1e-4;

    if !has_face_warp && !has_body_warp {
        return;
    }

    let (w, h) = (img.width, img.height);
    if w < 4 || h < 4 {
        return;
    }
    let src = img.clone();
    let src_data = &src.data;
    let (wf, hf) = (w as f32, h as f32);

    let sample_bilinear = |sx: f32, sy: f32| -> [f32; 3] {
        let fx = (sx - 0.5).clamp(0.0, (w - 1) as f32);
        let fy = (sy - 0.5).clamp(0.0, (h - 1) as f32);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let p00 = src_data.get(y0 * w + x0).copied().unwrap_or([0.0; 3]);
        let p10 = src_data.get(y0 * w + x1).copied().unwrap_or([0.0; 3]);
        let p01 = src_data.get(y1 * w + x0).copied().unwrap_or([0.0; 3]);
        let p11 = src_data.get(y1 * w + x1).copied().unwrap_or([0.0; 3]);
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let top = p00[c] + (p10[c] - p00[c]) * tx;
            let bot = p01[c] + (p11[c] - p01[c]) * tx;
            out[c] = top + (bot - top) * ty;
        }
        out
    };

    let (cx, cy, rx, ry) = (fz.cx, fz.cy, fz.rx, fz.ry);
    let eye_rx = (rx * 0.28).max(4.0);
    let eye_ry = (ry * 0.22).max(4.0);
    let (nose_x, nose_y) = fz.nose_tip;
    let nose_rx = (rx * 0.36).max(4.0);
    let nose_ry = (ry * 0.30).max(4.0);
    let chin_y = cy + ry * 0.82;
    let chin_rx = (rx * 0.72).max(4.0);
    let chin_ry = (ry * 0.48).max(4.0);
    let smile_rx = (rx * 0.28).max(4.0);
    let smile_ry = (ry * 0.26).max(4.0);

    // Anthropometric body proportions strictly anchored below the chin (adaptive to half-body, 3/4, or full-body shots).
    let chin_bottom = cy + ry * 0.96;
    let below_chin = (hf - chin_bottom).max(ry * 0.8);
    let body_h = (1.55 * ry).min(below_chin * 0.28).max(ry * 0.65).max(8.0);
    let body_rx = (rx * 2.25).clamp(wf * 0.18, wf * 0.46).max(8.0);
    let neck_y = chin_bottom + 0.22 * body_h;
    let shoulder_y = chin_bottom + 0.62 * body_h;
    let chest_y = chin_bottom + 1.25 * body_h;
    let arm_y = chin_bottom + 1.65 * body_h;
    let waist_y = chin_bottom + 2.15 * body_h;
    let hip_y = chin_bottom + 2.95 * body_h;
    let leg_y0 = chin_bottom + 2.80 * body_h;

    for_rows(&mut img.data, w, |y, row| {
        let py = y as f32 + 0.5;
        let v = (py - cy) / ry;
        if !has_body_warp && v.abs() > 1.50 {
            return;
        }
        if !has_face_warp && py < chin_bottom - ry * 0.35 {
            return;
        }

        for (x, px) in row.iter_mut().enumerate() {
            let px_x = x as f32 + 0.5;
            let u = (px_x - cx) / rx;
            let mut dx = 0.0f32;
            let mut dy = 0.0f32;

            // --- A. Facial Reshaping (when inside face envelope) ---
            if has_face_warp && v.abs() <= 1.50 && u.abs() <= 1.50 {
                // 1. Face & Jawline Slimming: pull lower-mid cheeks and jawline toward facial midline.
                if face_slim.abs() > 1e-4 {
                    let side = u.abs();
                    let w_jaw = bell2((side - 0.64) / 0.62, (v - 0.28) / 0.72);
                    dx += u.signum() * face_slim * (rx * 0.26) * w_jaw;
                    dy += face_slim * (ry * 0.055) * w_jaw;
                }

                // 2. Eye Size Enlargement: radial zoom around exact neural right and left eye landmarks.
                if eye_enlarge.abs() > 1e-4 {
                    for (ex, ey) in [fz.right_eye, fz.left_eye] {
                        let eu = (px_x - ex) / eye_rx;
                        let ev = (py - ey) / eye_ry;
                        let w_eye = bell2(eu, ev);
                        if w_eye > 0.0 {
                            dx -= (px_x - ex) * (eye_enlarge * 0.65 * w_eye);
                            dy -= (py - ey) * (eye_enlarge * 0.65 * w_eye);
                        }
                    }
                }

                // 3. Nose Slimming: horizontal pinch around the neural nose tip & bridge.
                if nose_slim.abs() > 1e-4 {
                    let nu = (px_x - nose_x) / nose_rx;
                    let nv = (py - nose_y) / nose_ry;
                    let w_nose = bell2(nu, nv) * nu.abs().min(1.0);
                    dx += nu.signum() * nose_slim * (nose_rx * 0.58) * w_nose;
                }

                // 4. Chin Length: vertical elongation (+) or shortening (-) of the lower chin.
                if chin_length.abs() > 1e-4 {
                    let cu = (px_x - cx) / chin_rx;
                    let cv = (py - chin_y) / chin_ry;
                    let w_chin = bell2(cu, cv);
                    dy -= chin_length * (ry * 0.24) * w_chin;
                }

                // 5. Smile Lift: upward (+) or downward (-) shift at neural right and left mouth corners.
                if smile_lift.abs() > 1e-4 {
                    for (sign, (mx, my)) in [(-1.0f32, fz.mouth_right), (1.0f32, fz.mouth_left)] {
                        let mu = (px_x - mx) / smile_rx;
                        let mv = (py - my) / smile_ry;
                        let w_smile = bell2(mu, mv);
                        if w_smile > 0.0 {
                            dy += smile_lift * (ry * 0.18) * w_smile;
                            dx -= sign * smile_lift * (rx * 0.050) * w_smile;
                        }
                    }
                }
            }

            // --- B. Full-Body Reshaping (Torso, Waist, Stomach, Hips, Shoulders, Arms, Neck, Chest, Legs) ---
            if has_body_warp && py >= chin_bottom - ry * 0.25 {
                let u_body = (px_x - cx) / body_rx;
                let side_body = u_body.abs();

                if side_body <= 1.70 {
                    // 1. Overall Body Slimming: smooth lateral compression across torso, waist, and hips.
                    if body_slim.abs() > 1e-4 {
                        let v_torso = (py - (chin_bottom + 1.95 * body_h)) / (body_h * 1.85);
                        let w_body = bell2((side_body - 0.65) / 0.65, v_torso) * side_body.min(1.0);
                        dx += u_body.signum() * body_slim * (body_rx * 0.28) * w_body;
                    }

                    // 2. Waist & Stomach Slimming: concentrated inward cinch at the midriff/waistline.
                    if waist_slim.abs() > 1e-4 {
                        let v_waist = (py - waist_y) / (body_h * 0.92);
                        let w_waist = bell2((side_body - 0.60) / 0.58, v_waist) * side_body.min(1.0);
                        dx += u_body.signum() * waist_slim * (body_rx * 0.30) * w_waist;
                        dy += waist_slim * (body_h * 0.055) * w_waist;
                    }

                    // 3. Stomach Flatten (Tummy Tuck): pulls central & lateral abdomen inward and lifts lower belly upward.
                    if stomach_flat > 1e-4 {
                        let v_stom = (py - (waist_y + 0.18 * body_h)) / (body_h * 0.88);
                        let w_stom_center = bell2(u_body / 0.82, v_stom);
                        let w_stom_side = bell2((side_body - 0.50) / 0.52, v_stom) * side_body.min(1.0);
                        dx += u_body.signum() * stomach_flat * (body_rx * 0.25) * w_stom_side;
                        dy += stomach_flat * (body_h * 0.18) * w_stom_center;
                    }

                    // 4. Hips & S-Curve: + expands hourglass hips outward while cinching upper waist; - slims hips inward.
                    if hip_enhance.abs() > 1e-4 {
                        let v_hip = (py - hip_y) / (body_h * 0.90);
                        let w_hip = bell2((side_body - 0.70) / 0.55, v_hip) * side_body.min(1.0);
                        dx -= u_body.signum() * hip_enhance * (body_rx * 0.28) * w_hip;
                        if hip_enhance > 0.0 {
                            let v_upper_s = (py - (waist_y - 0.10 * body_h)) / (body_h * 0.68);
                            let w_s = bell2((side_body - 0.58) / 0.48, v_upper_s) * side_body.min(1.0);
                            dx += u_body.signum() * hip_enhance * (body_rx * 0.11) * w_s;
                        }
                    }

                    // 5. Shoulder Width & Posture: + broadens & squares shoulders; - narrows broad shoulders.
                    if shoulder_width.abs() > 1e-4 {
                        let v_sh = (py - shoulder_y) / (body_h * 0.68);
                        let w_sh = bell2((side_body - 0.80) / 0.55, v_sh) * side_body.min(1.0);
                        dx -= u_body.signum() * shoulder_width * (body_rx * 0.25) * w_sh;
                        dy += shoulder_width * (body_h * 0.075) * w_sh;
                    }

                    // 6. Arm Slimming: pinches outer left & right arm corridors toward the arm axes.
                    if arm_slim.abs() > 1e-4 {
                        for arm_sign in [-1.0f32, 1.0f32] {
                            let arm_cx = cx + arm_sign * (body_rx * 0.88);
                            let au = (px_x - arm_cx) / (body_rx * 0.36).max(4.0);
                            let av = (py - arm_y) / (body_h * 1.10);
                            let w_arm = bell2(au, av) * au.abs().min(1.0);
                            if w_arm > 0.0 {
                                dx += au.signum() * arm_slim * (body_rx * 0.26) * w_arm;
                            }
                        }
                    }

                    // 7. Neck Slimming & Lengthening: pinches neck sides inward and subtly elongates neck.
                    if neck_slim.abs() > 1e-4 {
                        let nu = (px_x - cx) / (rx * 0.94).max(4.0);
                        let nv = (py - neck_y) / (body_h * 0.52).max(4.0);
                        let w_neck = bell2(nu, nv);
                        if w_neck > 0.0 {
                            dx += nu.signum() * neck_slim * (rx * 0.28) * w_neck * nu.abs().min(1.0);
                            dy += neck_slim * (body_h * 0.085) * w_neck;
                        }
                    }

                    // 8. Chest / Bust Sculpt: radial volume zoom + subtle upward lift around left/right chest anchors.
                    if chest_enhance.abs() > 1e-4 {
                        let chest_rx = (body_rx * 0.44).max(4.0);
                        let chest_ry = (body_h * 0.58).max(4.0);
                        for c_sign in [-1.0f32, 1.0f32] {
                            let ccx = cx + c_sign * (body_rx * 0.34);
                            let cu = (px_x - ccx) / chest_rx;
                            let cv = (py - chest_y) / chest_ry;
                            let w_chest = bell2(cu, cv);
                            if w_chest > 0.0 {
                                dx -= (px_x - ccx) * (chest_enhance * 0.48 * w_chest);
                                dy -= (py - chest_y) * (chest_enhance * 0.40 * w_chest);
                                dy += chest_enhance.max(0.0) * (body_h * 0.075) * w_chest;
                            }
                        }
                    }

                    // 9. Leg Lengthening & Slimming: vertical stretch in lower body + lateral thigh/calf slimming.
                    if leg_length.abs() > 1e-4 && py > leg_y0 - body_h * 0.35 {
                        let leg_span = (hf - leg_y0).max(body_h * 0.8);
                        let t_leg = ((py - leg_y0) / leg_span).clamp(0.0, 1.0);
                        let w_leg_v = smoothstep(0.0, 0.35, t_leg) * (1.0 - 0.35 * smoothstep(0.75, 1.0, t_leg));
                        let w_leg_h = bell2(u_body / 1.18, 0.0);
                        dy -= leg_length * (leg_span * 0.25) * w_leg_v * w_leg_h;
                        let w_thigh = bell2((side_body - 0.52) / 0.50, 0.0) * w_leg_v * side_body.min(1.0);
                        dx += u_body.signum() * leg_length * (body_rx * 0.18) * w_thigh;
                    }
                }
            }

            if dx.abs() > 1e-4 || dy.abs() > 1e-4 {
                let sx = (px_x + dx).clamp(0.5, wf - 0.5);
                let sy = (py + dy).clamp(0.5, hf - 0.5);
                *px = sample_bilinear(sx, sy);
            }
        }
    });
}

/// Apply portrait, skin & body retouching in place on scene-linear Rec.2020 `img`.
pub fn apply(img: &mut Rgb32f, s: &DevelopSettings, px_per_long: f64) {
    if !s.section_enabled("retouch") || !s.retouch.is_active() || img.data.is_empty() {
        return;
    }
    let r = &s.retouch;
    let (w, h) = (img.width, img.height);
    let n = w * h;

    let smooth = clean_u01(r.skin_smooth);
    let texture = if r.skin_texture.is_finite() {
        (r.skin_texture / 100.0).clamp(0.0, 1.0) as f32
    } else {
        0.5
    };
    let tone_even = clean_u01(r.skin_tone_even);
    let shine = clean_u01(r.shine_control);
    let dark_circles = clean_u01(r.dark_circles);
    let wrinkle = clean_u01(r.wrinkle_reduce);
    let radiance = clean_s11(r.skin_radiance);
    let warmth = clean_s11(r.skin_warmth);
    let redness = clean_s11(r.skin_redness);
    let eye_brighten = clean_u01(r.eye_brighten);
    let eye_whiten = clean_u01(r.eye_whiten);
    let eye_clarity = clean_u01(r.eye_clarity);
    let brow_enhance = clean_s11(r.eyebrow_enhance);
    let teeth_whiten = clean_u01(r.teeth_whiten);
    let lip_enhance = clean_s11(r.lip_enhance);
    let lip_moisture = clean_u01(r.lip_moisture);
    let face_contour = clean_u01(r.face_contour);
    let stomach_flat = clean_u01(r.stomach_flat);
    let neck_slim = clean_s11(r.neck_slim);
    let chest_enhance = clean_s11(r.chest_enhance);
    let hair_luster = clean_u01(r.hair_luster);
    let stray_hair = clean_u01(r.stray_hair);

    let long = (px_per_long as f32).max(w.max(h) as f32).max(64.0);
    let mut l_plane = Plane::new(w, h);
    let mut a_plane = Plane::new(w, h);
    let mut b_plane = Plane::new(w, h);
    let mut mask_plane = Plane::new(w, h);

    // Evaluate Oklab planes and compute an exposure-normalized gain so under/over-exposed scene-linear
    // images still land squarely inside the perceptual skin/lip/teeth/sclera loci.
    let exp_ev = if s.light.exposure.is_finite() { s.light.exposure as f32 } else { 0.0 };
    let exp_cbrt = exp_ev.clamp(-4.0, 4.0).exp2().cbrt();
    let mut sum_l_all = 0.0f64;
    for i in 0..n {
        if let Some(&rgb) = img.data.get(i) {
            let lab = oklab_from_2020([rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]);
            if lab[0].is_finite() {
                sum_l_all += f64::from(lab[0]);
            }
            if let Some(v) = l_plane.data.get_mut(i) {
                *v = lab[0];
            }
            if let Some(v) = a_plane.data.get_mut(i) {
                *v = lab[1];
            }
            if let Some(v) = b_plane.data.get_mut(i) {
                *v = lab[2];
            }
        }
    }
    let mean_l = (sum_l_all / (n.max(1) as f64)) as f32 * exp_cbrt;
    let cbrt_gain = if mean_l > 1e-3 && mean_l < 0.45 {
        (exp_cbrt * (0.52 / mean_l)).clamp(0.5, 2.5)
    } else {
        exp_cbrt.clamp(0.5, 2.5)
    };
    for i in 0..n {
        let lab_eval = [
            l_plane.data.get(i).copied().unwrap_or(0.0) * cbrt_gain,
            a_plane.data.get(i).copied().unwrap_or(0.0) * cbrt_gain,
            b_plane.data.get(i).copied().unwrap_or(0.0) * cbrt_gain,
        ];
        if let Some(v) = mask_plane.data.get_mut(i) {
            *v = skin_weight(lab_eval);
        }
    }

    let mask_sigma = (long * 0.003).clamp(1.0, 8.0);
    let skin_mask = gaussian(&mask_plane, mask_sigma);
    let fz = FaceZone::detect(img, w, h, &skin_mask, &a_plane, &b_plane, &l_plane);

    // 3-band Frequency Separation on perceptual lightness L:
    // 1. Fine micro-pore texture band: high = L - gaussian(L, fine_sigma)
    // 2. Edge-aware smooth contour base: base = guided_fast(low_in, mid_sigma, eps) + wide Gaussian blend
    // 3. Medium blemish/pimple/blotch/wrinkle/eye-bag band: mid = low_in - base
    let need_fine = smooth > 0.0
        || wrinkle > 0.0
        || stomach_flat > 0.0
        || eye_clarity > 0.0
        || brow_enhance.abs() > 1e-4
        || teeth_whiten > 0.0
        || lip_moisture > 0.0
        || hair_luster > 0.0
        || stray_hair > 0.0;
    let need_base = smooth > 0.0
        || shine > 0.0
        || dark_circles > 0.0
        || wrinkle > 0.0
        || stomach_flat > 0.0
        || eye_brighten > 0.0
        || eye_clarity > 0.0
        || brow_enhance.abs() > 1e-4
        || lip_moisture > 0.0
        || hair_luster > 0.0
        || stray_hair > 0.0;

    let fine_sigma = (long * 0.0020).clamp(0.9, 4.2);
    let mid_sigma = (long * 0.032).clamp(6.5, 64.0);
    let low_in = if need_fine { gaussian(&l_plane, fine_sigma) } else { l_plane.clone() };
    let base_l = if need_base {
        let (guided, wide_gauss) = par_join(
            || guided_fast(&low_in, mid_sigma, 0.048),
            || gaussian(&low_in, mid_sigma),
        );
        let mut out = guided;
        for ((b, &g), &l) in out.data.iter_mut().zip(wide_gauss.data.iter()).zip(low_in.data.iter()) {
            let edge = (l - g).abs();
            let w_gauss = 0.80 * (1.0 - smoothstep(0.085, 0.18, edge));
            *b = *b * (1.0 - w_gauss) + g * w_gauss;
        }
        out
    } else {
        low_in.clone()
    };

    // Chroma frequency separation on skin-weighted Oklab (a, b): active when smoothing skin, evening tone, or lifting dark circles.
    let (even_a, even_b) = if smooth > 0.0 || tone_even > 0.0 || dark_circles > 0.0 {
        let chroma_sigma = (long * 0.026).clamp(4.0, 64.0);
        let wa = a_plane.zip_map(&skin_mask, |a, m| a * m);
        let wb = b_plane.zip_map(&skin_mask, |b, m| b * m);
        let ((ba, bb), bm) = par_join(
            || par_join(|| gaussian(&wa, chroma_sigma), || gaussian(&wb, chroma_sigma)),
            || gaussian(&skin_mask, chroma_sigma),
        );
        let ea = ba.zip_map(&bm, |num, den| if den > 1e-4 { num / den } else { 0.0 });
        let eb = bb.zip_map(&bm, |num, den| if den > 1e-4 { num / den } else { 0.0 });
        (Some(ea), Some(eb))
    } else {
        (None, None)
    };

    // Texture retention target when skin_smooth > 0:
    // 0% -> 0.04 (airbrushed porcelain); 50% (default) -> 0.28 (visibly smooth studio skin, soft pores); 100% -> 1.05 (crisp editorial pores).
    let tex_target = if texture <= 0.5 {
        0.04 + (texture / 0.5) * 0.24
    } else {
        0.28 + ((texture - 0.5) / 0.5) * 0.77
    };

    let l_data = &l_plane.data;
    let low_data = &low_in.data;
    let base_data = &base_l.data;
    let a_data = &a_plane.data;
    let b_data = &b_plane.data;
    let m_data = &skin_mask.data;
    let ea_data = even_a.as_ref().map(|p| p.data.as_slice());
    let eb_data = even_b.as_ref().map(|p| p.data.as_slice());

    let eye_rx = (fz.rx * 0.38).max(4.0);
    let eye_ry = (fz.ry * 0.28).max(4.0);
    let mouth_cx = (fz.mouth_right.0 + fz.mouth_left.0) * 0.5;
    let mouth_rx = ((fz.mouth_left.0 - fz.mouth_right.0).abs() * 0.90).max(fz.rx * 0.50).max(4.0);
    let mouth_ry = (fz.ry * 0.34).max(4.0);

    let chin_bottom = fz.cy + fz.ry * 0.96;
    let below_chin = (h as f32 - chin_bottom).max(fz.ry * 0.8);
    let body_h = (1.55 * fz.ry).min(below_chin * 0.28).max(fz.ry * 0.65).max(8.0);
    let body_rx = (fz.rx * 2.25).clamp(w as f32 * 0.18, w as f32 * 0.46).max(8.0);
    let collar_y = chin_bottom + 0.55 * body_h;
    let chest_y = chin_bottom + 1.25 * body_h;
    let stomach_y = chin_bottom + 2.25 * body_h;

    for_rows(&mut img.data, w, |y, row| {
        let row_off = y * w;
        let py = y as f32 + 0.5;
        let v_face = (py - fz.cy) / fz.ry;

        for (x, px) in row.iter_mut().enumerate() {
            let i = row_off + x;
            let px_x = x as f32 + 0.5;
            let u_face = (px_x - fz.cx) / fz.rx;
            let u_body = (px_x - fz.cx) / body_rx;

            let l0 = l_data.get(i).copied().unwrap_or(0.0);
            let low = low_data.get(i).copied().unwrap_or(l0);
            let base = base_data.get(i).copied().unwrap_or(low);
            let a0 = a_data.get(i).copied().unwrap_or(0.0);
            let b0 = b_data.get(i).copied().unwrap_or(0.0);
            let lab_eval = [l0 * cbrt_gain, a0 * cbrt_gain, b0 * cbrt_gain];

            let high = l0 - low;
            let mid = low - base;
            let detail = high.abs() + 0.5 * mid.abs();

            // Anatomical feature protection around eyes, brows, and lips so strong skin smoothing keeps eyes/lashes/lips razor-sharp:
            let feature_protect = if fz.neural {
                let p_er = bell2((px_x - fz.right_eye.0) / (eye_rx * 0.88), (py - fz.right_eye.1) / (eye_ry * 0.82));
                let p_el = bell2((px_x - fz.left_eye.0) / (eye_rx * 0.88), (py - fz.left_eye.1) / (eye_ry * 0.82));
                let p_mouth = bell2((px_x - mouth_cx) / (mouth_rx * 0.78), (py - fz.mouth_y) / (mouth_ry * 0.72));
                p_er.max(p_el).max(p_mouth * 0.85)
            } else {
                0.0
            };

            // When a neural face is detected, cover both the face ellipse AND the full body corridor below the face.
            let subject_env = if fz.neural {
                let face_env = bell2(u_face / 1.95, (v_face - 0.25) / 2.15);
                let body_env = if py >= fz.cy {
                    bell2(u_body / 1.55, (py - (chin_bottom + 1.8 * body_h)) / (body_h * 3.2))
                } else {
                    0.0
                };
                face_env.max(body_env).max(0.35)
            } else {
                1.0
            };
            // Spatial face-skin floor so studio color casts, pale/cool skin, or shadowed faces still receive full retouching:
            let face_core = bell2(u_face / 1.02, (v_face - 0.04) / 1.12);
            let l_mid = smoothstep(0.08, 0.20, lab_eval[0]) * (1.0 - smoothstep(0.92, 0.995, lab_eval[0]));
            let non_feature = (1.0 - 0.80 * feature_protect) * (1.0 - 0.65 * smoothstep(0.030, 0.085, detail));
            let spatial_skin = if fz.neural {
                0.85 * face_core * l_mid * non_feature
            } else {
                0.55 * face_core * l_mid * non_feature
            };
            let w_skin = (m_data.get(i).copied().unwrap_or(0.0) * subject_env)
                .max(spatial_skin)
                .clamp(0.0, 1.0);
            let w_skin_smooth = (w_skin * (1.0 - 0.75 * feature_protect)).clamp(0.0, 1.0);

            let mut l = l0;
            let mut a = a0;
            let mut b = b0;
            let mut changed = false;

            // --- 1. Skin Retouching & Tone (Face + Full Body Skin) ---
            if w_skin > 1e-4 {
                if smooth > 0.0 {
                    let s_eff = smooth.powf(0.72);
                    let mid_atten = (1.0 - s_eff * 0.96 * w_skin_smooth).clamp(0.02, 1.0);
                    // Suppress isolated high-frequency blemish/pore spikes while keeping uniform micro-texture at tex_target:
                    let spike_damp = 1.0 - s_eff * 0.82 * smoothstep(0.008, 0.032, high.abs()) * w_skin_smooth;
                    let high_scale = ((1.0 + (tex_target - 1.0) * s_eff * w_skin_smooth) * spike_damp).clamp(0.03, 1.35);
                    l = base + mid * mid_atten + high * high_scale;
                    // Also soften red/brown blemish chroma spikes proportionally to skin_smooth:
                    if let (Some(eas), Some(ebs)) = (ea_data, eb_data) {
                        let ta = eas.get(i).copied().unwrap_or(a);
                        let tb = ebs.get(i).copied().unwrap_or(b);
                        let kc = (s_eff * 0.75 * w_skin_smooth).clamp(0.0, 0.88);
                        if ta.abs() > 1e-5 || tb.abs() > 1e-5 {
                            a += (ta - a) * kc;
                            b += (tb - b) * kc;
                        }
                    }
                    changed = true;
                }

                // Wrinkle & body crease/cellulite reduction: attenuate dark linear creases in both `mid` and `high`
                if wrinkle > 0.0 && (mid < 0.0 || high < 0.0) {
                    let crease_w = smoothstep(0.0008, 0.025, -(mid.min(0.0) + 0.6 * high.min(0.0))) * w_skin_smooth;
                    let cur_mid = l - base - high;
                    if cur_mid < 0.0 {
                        l -= cur_mid * (wrinkle * 0.96 * crease_w).clamp(0.0, 0.98);
                        changed = true;
                    }
                    if high < 0.0 {
                        l -= high * (wrinkle * 0.78 * crease_w).clamp(0.0, 0.92);
                        changed = true;
                    }
                }

                // Under-eye dark circles & eye bags: infraorbital zone right below neural eye landmarks + trough & shadow lift
                if dark_circles > 0.0 {
                    let dc_r = bell2((px_x - fz.right_eye.0) / (eye_rx * 1.25), (py - (fz.right_eye.1 + eye_ry * 0.68)) / (eye_ry * 1.15));
                    let dc_l = bell2((px_x - fz.left_eye.0) / (eye_rx * 1.25), (py - (fz.left_eye.1 + eye_ry * 0.68)) / (eye_ry * 1.15));
                    let under_eye_zone = dc_r
                        .max(dc_l)
                        .max(0.55 * bell2(u_face / 0.90, (v_face - 0.02) / 0.50));
                    let trough = (base - low).max(0.0);
                    let w_dc = w_skin * (0.40 + 0.60 * under_eye_zone);
                    if w_dc > 1e-3 {
                        let broad_shadow = ((0.72 - l * cbrt_gain).clamp(0.0, 0.35) / cbrt_gain.max(0.5)) * under_eye_zone * 0.45;
                        l += (trough * 1.10 + broad_shadow) * (dark_circles * w_dc).clamp(0.0, 0.98);
                        changed = true;
                    }
                    if let (Some(eas), Some(ebs)) = (ea_data, eb_data) {
                        let ta = eas.get(i).copied().unwrap_or(a);
                        let tb = ebs.get(i).copied().unwrap_or(b);
                        let k = (dark_circles * 0.85 * w_dc).clamp(0.0, 1.0);
                        if ta.abs() > 1e-5 || tb.abs() > 1e-5 {
                            a += (ta - a) * k;
                            b += (tb.max(b) - b) * k;
                            changed = true;
                        }
                    }
                }

                // Chroma frequency-separation skin tone evening
                if tone_even > 0.0
                    && let (Some(eas), Some(ebs)) = (ea_data, eb_data)
                {
                    let ta = eas.get(i).copied().unwrap_or(a);
                    let tb = ebs.get(i).copied().unwrap_or(b);
                    let k = (tone_even * 0.92 * w_skin).clamp(0.0, 0.98);
                    if ta.abs() > 1e-5 || tb.abs() > 1e-5 {
                        a += (ta - a) * k;
                        b += (tb - b) * k;
                    }
                    // Also even out medium luminance blotches when tone_even is raised:
                    let cur_mid = l - base - high;
                    l -= cur_mid * (tone_even * 0.58 * w_skin_smooth).clamp(0.0, 0.80);
                    changed = true;
                }

                // Specular shine control
                if shine > 0.0 && l > base {
                    let hi_w = smoothstep(0.34, 0.66, lab_eval[0]) * w_skin;
                    let excess = (l - base).max(0.0);
                    let compressed = excess / (1.0 + shine * 6.5 * hi_w * (excess / 0.045).max(0.45));
                    let dl = (excess - compressed) * hi_w * shine;
                    l = (l - dl).max(base);
                    let chroma_boost = 1.0 + dl * 1.05;
                    a *= chroma_boost;
                    b *= chroma_boost;
                    changed = true;
                }

                // Skin Radiance (subsurface midtone glow with highlight rolloff)
                if radiance.abs() > 1e-4 {
                    let mid_tone = (4.0 * l.clamp(0.0, 1.0) * (1.0 - l.clamp(0.0, 1.0))).clamp(0.18, 1.0);
                    l += radiance * 0.16 * mid_tone * w_skin;
                    changed = true;
                }

                // Skin Warmth (golden/tan shift along Oklab b and subtle a)
                if warmth.abs() > 1e-4 {
                    b += warmth * 0.052 * w_skin;
                    a += warmth * 0.012 * w_skin;
                    changed = true;
                }

                // Skin Redness (calms rosacea/flushing when negative, adds rosy tint when positive)
                if redness.abs() > 1e-4 {
                    let red_boost = if redness < 0.0 {
                        1.0 + smoothstep(0.02, 0.08, lab_eval[1]) * 0.85
                    } else {
                        1.0
                    };
                    a += redness * 0.048 * red_boost * w_skin;
                    changed = true;
                }

                // 3D Face Sculpt (dodge & burn contouring across T-zone, cheekbones, and jawline)
                if face_contour > 0.0 {
                    let t_zone_hi = bell2(u_face / 0.24, (v_face + 0.08) / 0.44)
                        .max(0.85 * bell2(u_face / 0.44, (v_face + 0.52) / 0.30))
                        .max(0.80 * bell2((u_face.abs() - 0.38) / 0.26, (v_face - 0.08) / 0.20))
                        .max(0.70 * bell2(u_face / 0.26, (v_face - 0.68) / 0.20));
                    let jaw_hollow = bell2((u_face.abs() - 0.70) / 0.35, (v_face - 0.34) / 0.45);
                    let sculpt = (t_zone_hi * 0.115 - jaw_hollow * 0.105) * face_contour * w_skin;
                    l += sculpt;
                    changed = true;
                }

                // 3D Collarbone & Neck Sculpt (subtle clavicle highlight + lateral neck shadow when neck_slim > 0)
                if neck_slim > 0.0 && py >= chin_bottom {
                    let w_clav = bell2((u_body.abs() - 0.38) / 0.28, (py - collar_y) / (body_h * 0.28));
                    if w_clav > 1e-3 {
                        l += 0.075 * neck_slim * w_clav * w_skin;
                        changed = true;
                    }
                }

                // 3D Chest Contour (subtle upper-chest highlight/shadow sculpt when chest_enhance > 0)
                if chest_enhance > 0.0 && py >= collar_y {
                    let w_c_hi = bell2((u_body.abs() - 0.30) / 0.28, (py - chest_y) / (body_h * 0.46));
                    let w_c_sh = bell2(u_body / 0.16, (py - chest_y) / (body_h * 0.42));
                    let c_sculpt = (w_c_hi * 0.085 - w_c_sh * 0.060) * chest_enhance * w_skin;
                    if c_sculpt.abs() > 1e-4 {
                        l += c_sculpt;
                        changed = true;
                    }
                }
            }

            // --- 1b. 3D Stomach Flatten (works on both bare skin and clothed abdomen) ---
            if stomach_flat > 0.0 && py >= chin_bottom + body_h * 0.8 {
                let w_ab = bell2(u_body / 0.82, (py - stomach_y) / (body_h * 0.92));
                if w_ab > 1e-3 {
                    // Compress convex mid-frequency bulge highlights and lift crease shadows across the abdomen:
                    let bulge = l - base - high;
                    l -= bulge * (stomach_flat * 0.92 * w_ab).clamp(0.0, 0.96);
                    // Subtle vertical abdominal line contour
                    let ab_side = bell2((u_body.abs() - 0.38) / 0.26, (py - stomach_y) / (body_h * 0.82));
                    l -= 0.055 * stomach_flat * ab_side;
                    changed = true;
                }
            }

            // --- 2. Eyes & Eyebrows ---
            if eye_brighten > 0.0 || eye_whiten > 0.0 || eye_clarity > 0.0 || brow_enhance.abs() > 1e-4 {
                let er = bell2((px_x - fz.right_eye.0) / eye_rx, (py - fz.right_eye.1) / eye_ry);
                let el = bell2((px_x - fz.left_eye.0) / eye_rx, (py - fz.left_eye.1) / eye_ry);
                let eye_box = er.max(el).max(0.45 * bell2(u_face / 0.88, (v_face + 0.16) / 0.44));
                if eye_box > 1e-3 {
                    let w_sclera = sclera_weight(lab_eval, m_data.get(i).copied().unwrap_or(0.0)) * eye_box;
                    if eye_whiten > 0.0 && w_sclera > 1e-3 {
                        let kw = (eye_whiten * w_sclera).clamp(0.0, 1.0);
                        l += (1.0 - l) * 0.30 * kw;
                        a *= 1.0 - 0.88 * kw;
                        b *= 1.0 - 0.88 * kw;
                        changed = true;
                    }

                    let w_iris = eye_box * (1.0 - 0.55 * m_data.get(i).copied().unwrap_or(0.0)) * (0.30 + 0.70 * smoothstep(0.0015, 0.022, detail));
                    if eye_brighten > 0.0 && w_iris > 1e-3 {
                        l += eye_brighten * 0.22 * w_iris * (1.0 - 0.35 * l);
                        a *= 1.0 + 0.24 * eye_brighten * w_iris;
                        b *= 1.0 + 0.24 * eye_brighten * w_iris;
                        changed = true;
                    }
                    if eye_clarity > 0.0 && w_iris > 1e-3 {
                        l += (high * 1.85 + mid * 0.95) * eye_clarity * w_iris;
                        changed = true;
                    }
                }

                if brow_enhance.abs() > 1e-4 {
                    let br = bell2((px_x - fz.right_eye.0) / (eye_rx * 1.25), (py - (fz.right_eye.1 - eye_ry * 0.68)) / (eye_ry * 0.95));
                    let bl = bell2((px_x - fz.left_eye.0) / (eye_rx * 1.25), (py - (fz.left_eye.1 - eye_ry * 0.68)) / (eye_ry * 0.95));
                    let brow_box = br.max(bl).max(0.45 * bell2(u_face / 0.88, (v_face + 0.32) / 0.34));
                    let dark_brow = (1.0 - smoothstep(0.45, 0.72, lab_eval[0])) * (0.30 + 0.70 * smoothstep(0.0015, 0.020, detail));
                    let w_brow = brow_box * dark_brow;
                    if w_brow > 1e-3 {
                        // Positive eyebrow_enhance deepens/darkens brows and adds crispness.
                        l -= brow_enhance * 0.18 * w_brow;
                        l += high * brow_enhance.max(0.0) * 1.05 * w_brow;
                        changed = true;
                    }
                }
            }

            // --- 3. Mouth, Teeth & Lips ---
            if teeth_whiten > 0.0 || lip_enhance.abs() > 1e-4 || lip_moisture > 0.0 {
                let mouth_u = (px_x - mouth_cx) / mouth_rx;
                let mouth_v = (py - fz.mouth_y) / mouth_ry;
                let mouth_box = bell2(mouth_u, mouth_v).max(0.55 * bell2(u_face / 0.92, (v_face - 0.42) / 0.58));
                if mouth_box > 1e-3 {
                    if teeth_whiten > 0.0 {
                        let w_teeth = teeth_weight(lab_eval, detail) * mouth_box;
                        if w_teeth > 1e-3 {
                            let kt = (teeth_whiten * w_teeth).clamp(0.0, 1.0);
                            l += (1.0 - l) * 0.32 * kt;
                            if b > 0.0 {
                                b *= 1.0 - 0.92 * kt;
                            }
                            a *= 1.0 - 0.65 * kt;
                            changed = true;
                        }
                    }

                    if lip_enhance.abs() > 1e-4 || lip_moisture > 0.0 {
                        let w_lip = lip_weight(lab_eval) * mouth_box;
                        if w_lip > 1e-3 {
                            if lip_enhance.abs() > 1e-4 {
                                let kl = lip_enhance * w_lip;
                                a += 0.022 * kl;
                                a *= 1.0 + 0.72 * kl;
                                b *= 1.0 + 0.28 * kl;
                                l -= 0.042 * kl;
                                changed = true;
                            }
                            if lip_moisture > 0.0 {
                                let sheen = (high.max(0.0) * 2.40 + mid.max(0.0) * 1.15 + 0.045 * smoothstep(0.42, 0.70, lab_eval[0])) * lip_moisture * w_lip;
                                l += sheen;
                                changed = true;
                            }
                        }
                    }
                }
            }

            // --- 4. Hair Luster & Stray Hair / Flyaway Reduction ---
            if hair_luster > 0.0 || stray_hair > 0.0 {
                let hair_arch = smoothstep(-1.65, -0.10, -v_face) * (1.0 - 0.75 * m_data.get(i).copied().unwrap_or(0.0));
                if hair_arch > 1e-3 {
                    if hair_luster > 0.0 {
                        let w_hair = hair_arch * (1.0 - smoothstep(0.68, 0.90, lab_eval[0])) * (0.25 + 0.75 * smoothstep(0.002, 0.024, detail));
                        if w_hair > 1e-3 {
                            let sheen = mid.max(0.0) * 1.15 + high * 0.90;
                            l += (sheen + 0.080 * l0) * hair_luster * w_hair;
                            changed = true;
                        }
                    }
                    if stray_hair > 0.0 {
                        // Thin isolated flyaway strands produce sharp high-frequency spikes (`high`) over a smooth guided base.
                        let spike = (high.abs() - 0.35 * mid.abs()).max(0.0);
                        let w_fly = hair_arch * smoothstep(0.0025, 0.025, spike);
                        if w_fly > 1e-3 {
                            l -= high * (stray_hair * 0.94 * w_fly).clamp(0.0, 0.98);
                            changed = true;
                        }
                    }
                }
            }

            if changed {
                let rgb = oklab_to_2020([l.clamp(0.0, 1.5), a, b]);
                *px = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
            }
        }
    });

    // Apply 2D facial reshaping / liquify mesh warp when any reshape slider is active.
    if r.has_reshape() {
        apply_reshape_warp(img, r, &fz);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skin_patch(w: usize, h: usize) -> Rgb32f {
        // Warm mid-tone skin in linear Rec.2020 with a medium-frequency blotch and a specular highlight.
        let base_lab = [0.68f32, 0.045, 0.055];
        let base_rgb = oklab_to_2020(base_lab);
        let mut img = Rgb32f::filled(w, h, base_rgb);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - (w / 2) as f32;
                let dy = y as f32 - (h / 2) as f32;
                let r2 = dx * dx + dy * dy;
                let mut lab = base_lab;
                // Medium-frequency luminance & redness blotch near center
                if r2 < 36.0 {
                    lab[0] -= 0.06;
                    lab[1] += 0.025;
                }
                // Bright specular shine spot in the upper-left quadrant
                let sx = x as f32 - (w / 4) as f32;
                let sy = y as f32 - (h / 4) as f32;
                if sx * sx + sy * sy < 16.0 {
                    lab[0] = 0.84;
                }
                img.data[y * w + x] = oklab_to_2020(lab);
            }
        }
        img
    }

    #[test]
    fn skin_weight_selects_skin_and_rejects_sky_or_foliage() {
        let skin = [0.68, 0.045, 0.055];
        let deep_skin = [0.42, 0.050, 0.060];
        let sky = [0.70, -0.03, -0.10];
        let foliage = [0.55, -0.08, 0.09];
        assert!(skin_weight(skin) > 0.7, "{}", skin_weight(skin));
        assert!(skin_weight(deep_skin) > 0.7, "{}", skin_weight(deep_skin));
        assert_eq!(skin_weight(sky), 0.0);
        assert_eq!(skin_weight(foliage), 0.0);
    }

    #[test]
    fn retouch_smooths_blotches_evens_tone_and_compresses_shine() {
        let orig = skin_patch(64, 64);
        let mut img = orig.clone();
        let mut s = DevelopSettings::default();
        s.retouch.skin_smooth = 75.0;
        s.retouch.skin_tone_even = 70.0;
        s.retouch.shine_control = 80.0;
        apply(&mut img, &s, 64.0);

        // Center blotch should be closer to surrounding skin lightness and redness than before.
        let c_idx = 32 * 64 + 32;
        let bg_idx = 32 * 64 + 52;
        let lab_before = oklab_from_2020(orig.data[c_idx]);
        let lab_after = oklab_from_2020(img.data[c_idx]);
        let lab_bg = oklab_from_2020(img.data[bg_idx]);
        assert!((lab_after[0] - lab_bg[0]).abs() < (lab_before[0] - lab_bg[0]).abs());
        assert!((lab_after[1] - lab_bg[1]).abs() < (lab_before[1] - lab_bg[1]).abs());

        // Specular highlight in upper-left should be compressed downward.
        let s_idx = 16 * 64 + 16;
        let shine_before = oklab_from_2020(orig.data[s_idx])[0];
        let shine_after = oklab_from_2020(img.data[s_idx])[0];
        assert!(shine_after < shine_before, "{shine_after} vs {shine_before}");
    }

    #[test]
    fn portrait_retouch_features_modify_skin_eyes_teeth_lips_and_reshape() {
        let orig = skin_patch(64, 64);
        let mut img = orig.clone();
        let mut s = DevelopSettings::default();
        s.retouch.dark_circles = 80.0;
        s.retouch.wrinkle_reduce = 70.0;
        s.retouch.skin_radiance = 40.0;
        s.retouch.skin_warmth = 30.0;
        s.retouch.skin_redness = -30.0;
        s.retouch.face_contour = 50.0;
        s.retouch.face_slim = 40.0;
        s.retouch.eye_enlarge = 35.0;
        s.retouch.nose_slim = 30.0;
        s.retouch.chin_length = -25.0;
        s.retouch.smile_lift = 30.0;
        apply(&mut img, &s, 64.0);

        let c_idx = 32 * 64 + 32;
        let lab_before = oklab_from_2020(orig.data[c_idx]);
        let lab_after = oklab_from_2020(img.data[c_idx]);
        // Dark circles + radiance should lift the dark center trough, warmth increases b, negative redness lowers a.
        assert!(lab_after[0] > lab_before[0], "{} vs {}", lab_after[0], lab_before[0]);
        assert!(lab_after[2] > lab_before[2], "{} vs {}", lab_after[2], lab_before[2]);
        assert!(lab_after[1] < lab_before[1], "{} vs {}", lab_after[1], lab_before[1]);
        assert!(lip_weight([0.52, 0.085, 0.030]) > 0.7);
        assert!(teeth_weight([0.82, 0.005, 0.020], 0.02) > 0.7);
        assert!(sclera_weight([0.78, 0.008, 0.010], 0.0) > 0.7);
    }

    #[test]
    fn retouch_smooths_blemishes_and_pores_on_steep_3d_shaded_face() {
        let (w, h) = (128, 128);
        let mut orig = Rgb32f::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // Steep 3D directional lighting ramp across the face + micro-pore roughness + blemish at center
                let ramp = 0.42 + 0.36 * (x as f32 / w as f32);
                let pore = if (x + y) % 2 == 1 { 0.018 } else { -0.018 };
                let dx = x as f32 - 64.0;
                let dy = y as f32 - 64.0;
                let blemish = if dx * dx + dy * dy < 25.0 { -0.065 } else { 0.0 };
                let lab = [ramp + pore + blemish, 0.042, 0.052];
                orig.data[y * w + x] = oklab_to_2020(lab);
            }
        }
        let mut img = orig.clone();
        let mut s = DevelopSettings::default();
        s.retouch.skin_smooth = 75.0; // default skin_texture = 50.0
        apply(&mut img, &s, 128.0);

        let c_idx = 64 * w + 64;
        let l_before = oklab_from_2020(orig.data[c_idx])[0];
        let l_after = oklab_from_2020(img.data[c_idx])[0];
        // The dark blemish + pore dip on the 3D-lit cheek must be strongly smoothed toward the local ramp (~0.60)
        assert!(l_after - l_before > 0.035, "expected strong smoothing on 3D ramp: before={l_before}, after={l_after}");
    }
}
