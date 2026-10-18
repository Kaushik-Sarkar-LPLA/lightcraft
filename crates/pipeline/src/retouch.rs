//! Portrait & skin retouching in scene-linear Rec.2020 (via perceptual Oklab):
//! - **Frequency-separation skin smoothing** (`skin_smooth`, `skin_texture`): separates perceptual
//!   lightness `L` into a fine pore/texture band (`L - G_fine(L)`), a medium blotch/bump band, and
//!   an edge-preserving guided-filter base layer (`guided_fast`), attenuating medium-frequency
//!   skin unevenness while preserving (or adjusting) fine pore texture.
//! - **Chroma frequency-separation tone evening** (`skin_tone_even`): evens out localized skin
//!   redness and colour blotchiness toward the local skin-weighted Oklab `(a, b)` mean.
//! - **Specular shine compression** (`shine_control`): compresses bright specular hot-spots on
//!   forehead, nose bridge, and cheeks back toward the local skin base luminance.

use lightcraft_color::perceptual::{oklab_from_2020, oklab_to_2020};
use lightcraft_develop::DevelopSettings;
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

/// Apply portrait & skin retouching in place on scene-linear Rec.2020 `img`.
pub fn apply(img: &mut Rgb32f, s: &DevelopSettings, px_per_long: f32) {
    if !s.section_enabled("retouch") || !s.retouch.is_active() || img.data.is_empty() {
        return;
    }
    let (w, h) = (img.width, img.height);
    let n = w * h;
    let smooth = if s.retouch.skin_smooth.is_finite() {
        (s.retouch.skin_smooth / 100.0).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let texture = if s.retouch.skin_texture.is_finite() {
        (s.retouch.skin_texture / 100.0).clamp(0.0, 1.0) as f32
    } else {
        0.5
    };
    let tone_even = if s.retouch.skin_tone_even.is_finite() {
        (s.retouch.skin_tone_even / 100.0).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let shine = if s.retouch.shine_control.is_finite() {
        (s.retouch.shine_control / 100.0).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    let long = px_per_long.max(w.max(h) as f32).max(64.0);
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

    // Soften the skin mask slightly so transitions at pores, blemishes, and specular highlights stay smooth.
    let mask_sigma = (long * 0.003).clamp(1.0, 8.0);
    let skin_mask = gaussian(&mask_plane, mask_sigma);

    // Frequency separation on perceptual lightness L:
    // 1. Fine pore texture band: high = L - gaussian(L, fine_sigma)
    // 2. Edge-aware base layer: base = guided_fast(low_in, mid_sigma, eps)
    // 3. Medium blotch/bump band: mid = low_in - base
    let fine_sigma = (long * 0.0018).clamp(0.8, 6.0);
    let mid_sigma = (long * 0.014).clamp(2.5, 42.0);
    let low_in = if smooth > 0.0 { gaussian(&l_plane, fine_sigma) } else { l_plane.clone() };
    let base_l = if smooth > 0.0 || shine > 0.0 {
        guided_fast(&low_in, mid_sigma, 0.0012)
    } else {
        low_in.clone()
    };

    // Chroma frequency separation on skin-weighted Oklab (a, b):
    let (even_a, even_b) = if tone_even > 0.0 {
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
        for (x, px) in row.iter_mut().enumerate() {
            let i = row_off + x;
            let w_skin = m_data.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            if w_skin <= 1e-4 {
                continue;
            }
            let l0 = l_data.get(i).copied().unwrap_or(0.0);
            let low = low_data.get(i).copied().unwrap_or(l0);
            let base = base_data.get(i).copied().unwrap_or(low);
            let mut l = l0;

            if smooth > 0.0 {
                let high = l0 - low;
                let mid = low - base;
                let mid_atten = 1.0 - smooth * 0.82 * w_skin;
                let high_scale = 1.0 + (tex_gain - 1.0) * smooth.max(0.35) * w_skin;
                l = base + mid * mid_atten + high * high_scale;
            }

            let mut a = a_data.get(i).copied().unwrap_or(0.0);
            let mut b = b_data.get(i).copied().unwrap_or(0.0);

            if tone_even > 0.0
                && let (Some(eas), Some(ebs)) = (ea_data, eb_data)
            {
                let ta = eas.get(i).copied().unwrap_or(a);
                let tb = ebs.get(i).copied().unwrap_or(b);
                let k = (tone_even * 0.75 * w_skin).clamp(0.0, 1.0);
                a += (ta - a) * k;
                b += (tb - b) * k;
            }

            if shine > 0.0 && l > base {
                // Compress localized specular highlights on bright skin while restoring a hint of skin chroma.
                let hi_w = smoothstep(0.50, 0.78, l0) * w_skin;
                let excess = (l - base).max(0.0);
                let compressed = excess / (1.0 + shine * 3.2 * hi_w * (excess / 0.08).max(0.25));
                let dl = (excess - compressed) * hi_w * shine;
                l = (l - dl).max(base);
                let chroma_boost = 1.0 + dl * 0.65;
                a *= chroma_boost;
                b *= chroma_boost;
            }

            let rgb = oklab_to_2020([l.max(0.0), a, b]);
            *px = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
        }
    });
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
}
