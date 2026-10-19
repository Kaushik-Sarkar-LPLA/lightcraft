//! Evoto-style portrait & skin retouching in scene-linear Rec.2020 (via perceptual Oklab):
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

/// Smooth perceptual skin-locus weight in Oklab (`0.0..=1.0`) covering Fitzpatrick types I–VI.
#[inline]
pub fn skin_weight(lab: [f32; 3]) -> f32 {
    let [l, a, b] = lab;
    if !l.is_finite() || !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    // Skin tones are warm (both a > 0 and b > 0) with moderate chroma and non-extreme lightness.
    let c = a.hypot(b);
    let hue = b.atan2(a); // radians; human skin clusters around 0.45..1.20 rad (~26°..69°)
    let w_l = smoothstep(0.16, 0.28, l) * (1.0 - smoothstep(0.88, 0.98, l));
    let w_c = smoothstep(0.012, 0.028, c) * (1.0 - smoothstep(0.16, 0.24, c));
    let w_h = smoothstep(0.32, 0.52, hue) * (1.0 - smoothstep(1.12, 1.32, hue));
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
    let w_l = smoothstep(0.20, 0.30, l) * (1.0 - smoothstep(0.78, 0.88, l));
    let w_a = smoothstep(0.038, 0.068, a);
    let w_rose = smoothstep(-0.008, 0.020, a - b * 0.72);
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
    let w_l = smoothstep(0.54, 0.68, l) * (1.0 - smoothstep(0.96, 0.998, l));
    let w_c = 1.0 - smoothstep(0.032, 0.072, c);
    let w_a = 1.0 - smoothstep(0.018, 0.042, a.abs());
    let w_b = smoothstep(-0.015, -0.002, b) * (1.0 - smoothstep(0.045, 0.075, b));
    let w_edge = 0.45 + 0.55 * smoothstep(0.004, 0.035, detail);
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
    let w_l = smoothstep(0.48, 0.62, l) * (1.0 - smoothstep(0.95, 0.995, l));
    let w_c = 1.0 - smoothstep(0.028, 0.065, c);
    (w_l * w_c * (1.0 - 0.80 * w_skin)).clamp(0.0, 1.0)
}

/// Detected or canonical portrait face geometry in pixel coordinates.
#[derive(Clone, Copy, Debug)]
struct FaceZone {
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    eye_y: f32,
    eye_dx: f32,
    mouth_y: f32,
}

impl FaceZone {
    /// Estimate the primary face ellipse and feature anchors from the skin and lip masks,
    /// falling back smoothly to canonical portrait coordinates when skin coverage is sparse.
    fn detect(w: usize, h: usize, skin: &Plane, a_plane: &Plane, b_plane: &Plane, l_plane: &Plane) -> Self {
        let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
        let ( mut sum_w, mut sum_x, mut sum_y ) = (0.0f64, 0.0f64, 0.0f64);
        let step = (w.min(h) / 64).max(1);
        for y in (0..h).step_by(step) {
            let py = y as f32 + 0.5;
            // Bias slightly toward the upper 75% of the frame so neck/chest skin doesn't drag the face center down.
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
            let (cx, cy, rx, ry) = (wf * 0.5, hf * 0.44, wf * 0.26, hf * 0.32);
            return Self {
                cx,
                cy,
                rx: rx.max(8.0),
                ry: ry.max(8.0),
                eye_y: cy - ry * 0.16,
                eye_dx: rx * 0.38,
                mouth_y: cy + ry * 0.44,
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

        // Refine mouth vertical anchor if a clear lip cluster exists in the lower face.
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

        Self {
            cx,
            cy,
            rx,
            ry,
            eye_y: cy - ry * 0.16,
            eye_dx: rx * 0.38,
            mouth_y,
        }
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

/// Apply 2D C2-continuous inverse bilinear mesh warp for facial & feature reshaping
/// (`face_slim`, `eye_enlarge`, `nose_slim`, `chin_length`, `smile_lift`).
fn apply_reshape_warp(img: &mut Rgb32f, r: &Retouch, fz: &FaceZone) {
    let face_slim = clean_s11(r.face_slim);
    let eye_enlarge = clean_s11(r.eye_enlarge);
    let nose_slim = clean_s11(r.nose_slim);
    let chin_length = clean_s11(r.chin_length);
    let smile_lift = clean_s11(r.smile_lift);

    if face_slim.abs() <= 1e-4
        && eye_enlarge.abs() <= 1e-4
        && nose_slim.abs() <= 1e-4
        && chin_length.abs() <= 1e-4
        && smile_lift.abs() <= 1e-4
    {
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
    let eye_rx = (rx * 0.25).max(4.0);
    let eye_ry = (ry * 0.20).max(4.0);
    let nose_y = cy + ry * 0.06;
    let nose_rx = (rx * 0.34).max(4.0);
    let nose_ry = (ry * 0.28).max(4.0);
    let chin_y = cy + ry * 0.82;
    let chin_rx = (rx * 0.68).max(4.0);
    let chin_ry = (ry * 0.45).max(4.0);
    let mouth_dx = rx * 0.28;
    let smile_rx = (rx * 0.26).max(4.0);
    let smile_ry = (ry * 0.24).max(4.0);

    for_rows(&mut img.data, w, |y, row| {
        let py = y as f32 + 0.5;
        let v = (py - cy) / ry;
        if v.abs() > 1.45 {
            return;
        }
        for (x, px) in row.iter_mut().enumerate() {
            let px_x = x as f32 + 0.5;
            let u = (px_x - cx) / rx;
            if u.abs() > 1.45 {
                continue;
            }
            let mut dx = 0.0f32;
            let mut dy = 0.0f32;

            // 1. Face & Jawline Slimming: pull lower-mid cheeks and jawline toward facial midline.
            if face_slim.abs() > 1e-4 {
                let side = u.abs();
                let w_jaw = bell2((side - 0.68) / 0.52, (v - 0.32) / 0.62);
                // Positive face_slim samples further outward (sign(u)), moving the contour inward.
                dx += u.signum() * face_slim * (rx * 0.065) * w_jaw;
                dy += face_slim * (ry * 0.015) * w_jaw;
            }

            // 2. Eye Size Enlargement: radial zoom around left and right eye centers.
            if eye_enlarge.abs() > 1e-4 {
                for sign in [-1.0f32, 1.0f32] {
                    let ex = cx + sign * fz.eye_dx;
                    let ey = fz.eye_y;
                    let eu = (px_x - ex) / eye_rx;
                    let ev = (py - ey) / eye_ry;
                    let w_eye = bell2(eu, ev);
                    if w_eye > 0.0 {
                        dx -= (px_x - ex) * (eye_enlarge * 0.22 * w_eye);
                        dy -= (py - ey) * (eye_enlarge * 0.22 * w_eye);
                    }
                }
            }

            // 3. Nose Slimming: horizontal pinch around the nose bridge and alar wings.
            if nose_slim.abs() > 1e-4 {
                let nu = (px_x - cx) / nose_rx;
                let nv = (py - nose_y) / nose_ry;
                let w_nose = bell2(nu, nv) * nu.abs().min(1.0);
                dx += nu.signum() * nose_slim * (nose_rx * 0.16) * w_nose;
            }

            // 4. Chin Length: vertical elongation (+) or shortening (-) of the lower chin.
            if chin_length.abs() > 1e-4 {
                let cu = (px_x - cx) / chin_rx;
                let cv = (py - chin_y) / chin_ry;
                let w_chin = bell2(cu, cv);
                dy -= chin_length * (ry * 0.075) * w_chin;
            }

            // 5. Smile Lift: upward (+) or downward (-) shift at left and right mouth corners.
            if smile_lift.abs() > 1e-4 {
                for sign in [-1.0f32, 1.0f32] {
                    let mx = cx + sign * mouth_dx;
                    let my = fz.mouth_y;
                    let mu = (px_x - mx) / smile_rx;
                    let mv = (py - my) / smile_ry;
                    let w_smile = bell2(mu, mv);
                    if w_smile > 0.0 {
                        // Positive smile_lift samples slightly lower (+y), moving mouth corners upward (-y).
                        dy += smile_lift * (ry * 0.045) * w_smile;
                        dx -= sign * smile_lift * (rx * 0.015) * w_smile;
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

/// Apply Evoto-style portrait & skin retouching in place on scene-linear Rec.2020 `img`.
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
    let hair_luster = clean_u01(r.hair_luster);
    let stray_hair = clean_u01(r.stray_hair);

    let long = (px_per_long as f32).max(w.max(h) as f32).max(64.0);
    let mut l_plane = Plane::new(w, h);
    let mut a_plane = Plane::new(w, h);
    let mut b_plane = Plane::new(w, h);
    let mut mask_plane = Plane::new(w, h);

    for i in 0..n {
        if let Some(&rgb) = img.data.get(i) {
            let lab = oklab_from_2020([rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]);
            if let Some(v) = l_plane.data.get_mut(i) {
                *v = lab[0];
            }
            if let Some(v) = a_plane.data.get_mut(i) {
                *v = lab[1];
            }
            if let Some(v) = b_plane.data.get_mut(i) {
                *v = lab[2];
            }
            if let Some(v) = mask_plane.data.get_mut(i) {
                *v = skin_weight(lab);
            }
        }
    }

    let mask_sigma = (long * 0.003).clamp(1.0, 8.0);
    let skin_mask = gaussian(&mask_plane, mask_sigma);
    let fz = FaceZone::detect(w, h, &skin_mask, &a_plane, &b_plane, &l_plane);

    // Frequency separation on perceptual lightness L:
    // 1. Fine pore/strand texture band: high = L - gaussian(L, fine_sigma)
    // 2. Edge-aware base layer: base = guided_fast(low_in, mid_sigma, eps)
    // 3. Medium blotch/wrinkle/eye-bag band: mid = low_in - base
    let need_fine = smooth > 0.0
        || wrinkle > 0.0
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
        || eye_brighten > 0.0
        || eye_clarity > 0.0
        || brow_enhance.abs() > 1e-4
        || lip_moisture > 0.0
        || hair_luster > 0.0
        || stray_hair > 0.0;

    let fine_sigma = (long * 0.0018).clamp(0.8, 6.0);
    let mid_sigma = (long * 0.014).clamp(2.5, 42.0);
    let low_in = if need_fine { gaussian(&l_plane, fine_sigma) } else { l_plane.clone() };
    let base_l = if need_base {
        guided_fast(&low_in, mid_sigma, 0.0012)
    } else {
        low_in.clone()
    };

    // Chroma frequency separation on skin-weighted Oklab (a, b):
    let (even_a, even_b) = if tone_even > 0.0 || dark_circles > 0.0 {
        let chroma_sigma = (long * 0.018).clamp(3.0, 50.0);
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

    // Texture gain: 50% (0.5) preserves 100% of fine pore texture; 0% softens pores (0.35x); 100% enhances pores (1.45x).
    let tex_gain = if texture <= 0.5 {
        0.35 + (texture / 0.5) * 0.65
    } else {
        1.0 + ((texture - 0.5) / 0.5) * 0.45
    };

    let l_data = &l_plane.data;
    let low_data = &low_in.data;
    let base_data = &base_l.data;
    let a_data = &a_plane.data;
    let b_data = &b_plane.data;
    let m_data = &skin_mask.data;
    let ea_data = even_a.as_ref().map(|p| p.data.as_slice());
    let eb_data = even_b.as_ref().map(|p| p.data.as_slice());

    for_rows(&mut img.data, w, |y, row| {
        let row_off = y * w;
        let py = y as f32 + 0.5;
        let v_face = (py - fz.cy) / fz.ry;

        for (x, px) in row.iter_mut().enumerate() {
            let i = row_off + x;
            let px_x = x as f32 + 0.5;
            let u_face = (px_x - fz.cx) / fz.rx;

            let l0 = l_data.get(i).copied().unwrap_or(0.0);
            let low = low_data.get(i).copied().unwrap_or(l0);
            let base = base_data.get(i).copied().unwrap_or(low);
            let a0 = a_data.get(i).copied().unwrap_or(0.0);
            let b0 = b_data.get(i).copied().unwrap_or(0.0);
            let w_skin = m_data.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);

            let high = l0 - low;
            let mid = low - base;
            let detail = high.abs() + 0.5 * mid.abs();

            let mut l = l0;
            let mut a = a0;
            let mut b = b0;
            let mut changed = false;

            // --- 1. Skin Retouching & Tone ---
            if w_skin > 1e-4 {
                if smooth > 0.0 {
                    let mid_atten = 1.0 - smooth * 0.82 * w_skin;
                    let high_scale = 1.0 + (tex_gain - 1.0) * smooth.max(0.35) * w_skin;
                    l = base + mid * mid_atten + high * high_scale;
                    changed = true;
                }

                // Wrinkle reduction: attenuate narrow dark linear creases (mid < 0) while preserving pores (high)
                if wrinkle > 0.0 && mid < 0.0 {
                    let crease_w = smoothstep(0.002, 0.035, -mid) * w_skin;
                    let cur_mid = l - base - high;
                    if cur_mid < 0.0 {
                        l -= cur_mid * (wrinkle * 0.85 * crease_w).clamp(0.0, 0.92);
                        changed = true;
                    }
                }

                // Under-eye dark circles & eye bags: infraorbital zone + dark cool skin trough lift
                if dark_circles > 0.0 {
                    let under_eye_zone = bell2((u_face.abs() - 0.36) / 0.38, (v_face - 0.04) / 0.26)
                        .max(0.45 * bell2(u_face / 0.85, (v_face - 0.02) / 0.45));
                    let trough = (base - low).max(0.0);
                    let w_dc = w_skin * (0.35 + 0.65 * under_eye_zone);
                    if trough > 0.0 && w_dc > 1e-3 {
                        l += trough * (dark_circles * 0.88 * w_dc).clamp(0.0, 0.90);
                        changed = true;
                    }
                    if let (Some(eas), Some(ebs)) = (ea_data, eb_data) {
                        let ta = eas.get(i).copied().unwrap_or(a);
                        let tb = ebs.get(i).copied().unwrap_or(b);
                        let k = (dark_circles * 0.65 * w_dc).clamp(0.0, 1.0);
                        a += (ta - a) * k;
                        b += (tb.max(b) - b) * k;
                        changed = true;
                    }
                }

                // Chroma frequency-separation skin tone evening
                if tone_even > 0.0
                    && let (Some(eas), Some(ebs)) = (ea_data, eb_data)
                {
                    let ta = eas.get(i).copied().unwrap_or(a);
                    let tb = ebs.get(i).copied().unwrap_or(b);
                    let k = (tone_even * 0.75 * w_skin).clamp(0.0, 1.0);
                    a += (ta - a) * k;
                    b += (tb - b) * k;
                    changed = true;
                }

                // Specular shine control
                if shine > 0.0 && l > base {
                    let hi_w = smoothstep(0.50, 0.78, l0) * w_skin;
                    let excess = (l - base).max(0.0);
                    let compressed = excess / (1.0 + shine * 3.2 * hi_w * (excess / 0.08).max(0.25));
                    let dl = (excess - compressed) * hi_w * shine;
                    l = (l - dl).max(base);
                    let chroma_boost = 1.0 + dl * 0.65;
                    a *= chroma_boost;
                    b *= chroma_boost;
                    changed = true;
                }

                // Skin Radiance (subsurface midtone glow with highlight rolloff)
                if radiance.abs() > 1e-4 {
                    let mid_tone = (4.0 * l * (1.0 - l)).clamp(0.0, 1.0);
                    l += radiance * 0.075 * mid_tone * w_skin;
                    changed = true;
                }

                // Skin Warmth (golden/tan shift along Oklab b and subtle a)
                if warmth.abs() > 1e-4 {
                    b += warmth * 0.024 * w_skin;
                    a += warmth * 0.005 * w_skin;
                    changed = true;
                }

                // Skin Redness (calms rosacea/flushing when negative, adds rosy tint when positive)
                if redness.abs() > 1e-4 {
                    a += redness * 0.022 * w_skin;
                    changed = true;
                }

                // 3D Face Sculpt (dodge & burn contouring across T-zone, cheekbones, and jawline)
                if face_contour > 0.0 {
                    let t_zone_hi = bell2(u_face / 0.22, (v_face + 0.08) / 0.42)
                        .max(0.80 * bell2(u_face / 0.42, (v_face + 0.52) / 0.28))
                        .max(0.75 * bell2((u_face.abs() - 0.38) / 0.24, (v_face - 0.08) / 0.18))
                        .max(0.65 * bell2(u_face / 0.24, (v_face - 0.68) / 0.18));
                    let jaw_hollow = bell2((u_face.abs() - 0.72) / 0.32, (v_face - 0.34) / 0.42);
                    let sculpt = (t_zone_hi * 0.045 - jaw_hollow * 0.042) * face_contour * w_skin;
                    l += sculpt;
                    changed = true;
                }
            }

            // --- 2. Eyes & Eyebrows ---
            if eye_brighten > 0.0 || eye_whiten > 0.0 || eye_clarity > 0.0 || brow_enhance.abs() > 1e-4 {
                let eye_box = bell2((u_face.abs() - 0.38) / 0.36, (v_face + 0.16) / 0.26)
                    .max(0.35 * bell2(u_face / 0.85, (v_face + 0.16) / 0.40));
                if eye_box > 1e-3 {
                    let w_sclera = sclera_weight([l0, a0, b0], w_skin) * eye_box;
                    if eye_whiten > 0.0 && w_sclera > 1e-3 {
                        let kw = (eye_whiten * w_sclera).clamp(0.0, 1.0);
                        l += (1.0 - l) * 0.12 * kw;
                        a *= 1.0 - 0.70 * kw;
                        b *= 1.0 - 0.70 * kw;
                        changed = true;
                    }

                    let w_iris = eye_box * (1.0 - 0.75 * w_skin) * smoothstep(0.004, 0.030, detail);
                    if eye_brighten > 0.0 && w_iris > 1e-3 {
                        l += eye_brighten * 0.085 * w_iris * (1.0 - 0.5 * l);
                        changed = true;
                    }
                    if eye_clarity > 0.0 && w_iris > 1e-3 {
                        l += (high * 0.85 + mid * 0.45) * eye_clarity * w_iris;
                        changed = true;
                    }
                }

                if brow_enhance.abs() > 1e-4 {
                    let brow_box = bell2((u_face.abs() - 0.38) / 0.38, (v_face + 0.32) / 0.20)
                        .max(0.35 * bell2(u_face / 0.85, (v_face + 0.32) / 0.30));
                    let dark_brow = (1.0 - smoothstep(0.38, 0.62, l0)) * (1.0 - 0.65 * w_skin) * smoothstep(0.003, 0.025, detail);
                    let w_brow = brow_box * dark_brow;
                    if w_brow > 1e-3 {
                        // Positive eyebrow_enhance deepens/darkens brows and adds crispness.
                        l -= brow_enhance * 0.075 * w_brow;
                        l += high * brow_enhance.max(0.0) * 0.55 * w_brow;
                        changed = true;
                    }
                }
            }

            // --- 3. Mouth, Teeth & Lips ---
            if teeth_whiten > 0.0 || lip_enhance.abs() > 1e-4 || lip_moisture > 0.0 {
                let mouth_v = (py - fz.mouth_y) / (fz.ry * 0.32).max(4.0);
                let mouth_box = bell2(u_face / 0.52, mouth_v).max(0.45 * bell2(u_face / 0.90, (v_face - 0.42) / 0.55));
                if mouth_box > 1e-3 {
                    if teeth_whiten > 0.0 {
                        let w_teeth = teeth_weight([l0, a0, b0], detail) * mouth_box;
                        if w_teeth > 1e-3 {
                            let kt = (teeth_whiten * w_teeth).clamp(0.0, 1.0);
                            l += (1.0 - l) * 0.14 * kt;
                            if b > 0.0 {
                                b *= 1.0 - 0.78 * kt;
                            }
                            a *= 1.0 - 0.45 * kt;
                            changed = true;
                        }
                    }

                    if lip_enhance.abs() > 1e-4 || lip_moisture > 0.0 {
                        let w_lip = lip_weight([l0, a0, b0]) * mouth_box;
                        if w_lip > 1e-3 {
                            if lip_enhance.abs() > 1e-4 {
                                let kl = lip_enhance * w_lip;
                                a *= 1.0 + 0.45 * kl;
                                b *= 1.0 + 0.18 * kl;
                                l -= 0.022 * kl;
                                changed = true;
                            }
                            if lip_moisture > 0.0 {
                                let sheen = (high.max(0.0) * 1.35 + mid.max(0.0) * 0.55) * lip_moisture * w_lip;
                                l += sheen;
                                changed = true;
                            }
                        }
                    }
                }
            }

            // --- 4. Hair Luster & Stray Hair / Flyaway Reduction ---
            if hair_luster > 0.0 || stray_hair > 0.0 {
                let hair_arch = smoothstep(-1.55, -0.15, -v_face) * (1.0 - 0.85 * w_skin);
                if hair_arch > 1e-3 {
                    if hair_luster > 0.0 {
                        let w_hair = hair_arch * (1.0 - smoothstep(0.65, 0.88, l0)) * smoothstep(0.004, 0.028, detail);
                        if w_hair > 1e-3 {
                            let sheen = mid.max(0.0) * 0.55 + high * 0.45;
                            l += (sheen + 0.035 * l0) * hair_luster * w_hair;
                            changed = true;
                        }
                    }
                    if stray_hair > 0.0 {
                        // Thin isolated flyaway strands produce sharp high-frequency spikes (`high`) over a smooth guided base.
                        let spike = (high.abs() - 0.5 * mid.abs()).max(0.0);
                        let w_fly = hair_arch * smoothstep(0.005, 0.035, spike);
                        if w_fly > 1e-3 {
                            l -= high * (stray_hair * 0.82 * w_fly).clamp(0.0, 0.90);
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
    fn evoto_portrait_features_modify_skin_eyes_teeth_lips_and_reshape() {
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
}
