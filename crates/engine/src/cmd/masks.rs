//! Masking and Remove (spot) commands on the active photo.

use lightcraft_develop::{BrushStroke, LocalAdjustments, Mask, MaskComponent, MaskOp, MaskShape, RedEye, Spot, SpotMode};
use lightcraft_geom::Point;
use serde_json::{Value, json};

use super::{CommandSpec, bad, bool_or, cmd, f64_or, has_active, ok, point, str_param};
use crate::segment::{Click, MAX_CLICKS};
use crate::{Result, Session};

fn shape_from(kind: &str, p: &Value, c: &str) -> Result<MaskShape> {
    Ok(match kind {
        "brush" => MaskShape::Brush { strokes: vec![] },
        "linear" => {
            MaskShape::Linear { start: point(p, "start").unwrap_or(Point::new(0.5, 0.25)), end: point(p, "end").unwrap_or(Point::new(0.5, 0.6)) }
        }
        "radial" => MaskShape::Radial {
            center: point(p, "center").unwrap_or(Point::new(0.5, 0.5)),
            rx: f64_or(p, "rx", 0.22),
            ry: f64_or(p, "ry", 0.16),
            angle: f64_or(p, "angle", 0.0),
            feather: f64_or(p, "feather", 50.0),
            invert: bool_or(p, "invert", false),
        },
        "sky" => MaskShape::Sky,
        "subject" => MaskShape::Subject,
        "background" => MaskShape::Background,
        "people" => {
            let person = p.get("person").and_then(Value::as_u64).unwrap_or(0) as u32;
            let parts: Vec<String> = p
                .get("parts")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
                .unwrap_or_else(|| vec!["Skin".into()]);
            MaskShape::People { person, parts }
        }
        "luminanceRange" => MaskShape::LuminanceRange {
            lo: f64_or(p, "lo", 0.6),
            hi: f64_or(p, "hi", 1.0),
            lo_feather: f64_or(p, "loFeather", 0.1),
            hi_feather: f64_or(p, "hiFeather", 0.1),
        },
        "colorRange" => {
            let samples: Vec<[f64; 3]> = p
                .get("samples")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|s| Some([s.get(0)?.as_f64()?, s.get(1)?.as_f64()?, s.get(2)?.as_f64()?])).collect())
                .unwrap_or_default();
            MaskShape::ColorRange { samples, refine: f64_or(p, "refine", 50.0) }
        }
        "object" => MaskShape::Object { hint: points(p, "points"), exclude: points(p, "exclude"), seg: seg_param(p), detail: vec![], edge: 0.0 },
        "prompt" => {
            MaskShape::Prompt { text: str_param(p, "text").unwrap_or_default().trim().to_string(), seg: seg_param(p), detail: vec![], edge: 0.0 }
        }
        other => {
            return Err(bad(
                c,
                format!("unknown mask kind `{other}` (brush|linear|radial|sky|subject|background|people|luminanceRange|colorRange|object|prompt)"),
            ));
        }
    })
}

/// `[[x, y], …]` normalized points under `key`.
fn points(p: &Value, key: &str) -> Vec<Point> {
    p.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|q| Some(Point::new(q.get(0)?.as_f64()?, q.get(1)?.as_f64()?))).collect())
        .unwrap_or_default()
}

/// A segmentation passed in (`seg`: a stored or replayed AI mask; the model isn't needed).
fn seg_param(p: &Value) -> Option<lightcraft_develop::SegMask> {
    p.get("seg").and_then(|v| serde_json::from_value(v.clone()).ok())
}

/// What a new AI shape still needs once it is added.
enum Later {
    Nothing,
    /// Background mode: these Object clicks are segmented on the worker.
    Clicks(Vec<Point>, Vec<Point>),
    /// Background mode: the Describe selection is computed first; the mask is added when found.
    Text(String),
}

/// Compute the segmentation of a new AI shape (Object with clicks, Prompt) — or, in background
/// mode, say what to queue. An Object without clicks stays empty until the photo is clicked
/// (the photo is analyzed meanwhile); a Prompt that matches nothing is an error. Shapes given
/// with their `seg` need no model.
fn resolve_ai(s: &mut Session, shape: &mut MaskShape, c: &str) -> Result<Later> {
    let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
    let background = s.segmenter.background;
    match shape {
        MaskShape::Object { seg: Some(_), .. } | MaskShape::Prompt { seg: Some(_), .. } => {}
        MaskShape::Object { hint, exclude, .. } if !hint.is_empty() && background => {
            s.segmenter.require_model()?;
            return Ok(Later::Clicks(std::mem::take(hint), std::mem::take(exclude)));
        }
        MaskShape::Object { hint, exclude, seg, .. } if !hint.is_empty() => {
            *seg = Some(segment(s, |s| s.segment_clicks(id, &clicks_of(hint, exclude)))?);
        }
        MaskShape::Object { .. } => {
            // the clicks come later: refuse an Object mask that could never be computed (no
            // model) rather than leave an empty one; in the app, analyze the photo meanwhile
            s.segmenter.require_model()?;
            if background {
                segment(s, |s| s.segment_prepare(id))?;
            }
        }
        MaskShape::Prompt { text, .. } if background => {
            s.segmenter.require_model()?;
            if text.trim().is_empty() {
                return Err(ai_err("describe what to select"));
            }
            return Ok(Later::Text(text.clone()));
        }
        MaskShape::Prompt { text, seg, .. } => {
            let found = segment(s, |s| s.segment_text(id, text))?;
            *seg = Some(found.ok_or_else(|| ai_err(format!("nothing matching “{text}” was found in this photo")))?);
        }
        _ => {}
    }
    Ok(Later::Nothing)
}

/// Queue what a new AI component (component `comp` of mask `mask`) still needs.
fn queue_later(s: &mut Session, later: Later, mask: u32, comp: usize, mut out: Value) -> Result<Value> {
    if let Later::Clicks(hint, exclude) = later {
        let id = s.active().ok_or_else(|| ai_err("no active photo"))?;
        segment(s, |s| s.segment_clicks_later(id, mask, comp, hint, exclude))?;
        out["pending"] = json!(true);
    }
    Ok(out)
}

/// AI mask failures are shown as they are (not as "invalid parameters").
fn ai_err(msg: impl Into<String>) -> crate::EngineError {
    crate::EngineError::Other(msg.into())
}

/// A segmentation call: the model is checked first, so a missing one is [`crate::EngineError::ModelMissing`]
/// (the segmenter's own errors are text).
fn segment<T>(s: &mut Session, f: impl FnOnce(&mut Session) -> std::result::Result<T, String>) -> Result<T> {
    s.segmenter.require_model()?;
    f(s).map_err(ai_err)
}

pub(crate) fn clicks_of(include: &[Point], exclude: &[Point]) -> Vec<Click> {
    include.iter().map(|p| Click { at: *p, include: true }).chain(exclude.iter().map(|p| Click { at: *p, include: false })).collect()
}

fn masks_edit(s: &mut Session, c: &str, label: &str, f: impl FnOnce(&mut Vec<Mask>, &mut Option<u32>) -> Result<()>) -> Result<Value> {
    let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
    let mut d = (*s.develop_of(id).unwrap_or_default()).clone();
    let mut active = s.active_mask;
    f(&mut d.masks, &mut active)?;
    s.set_develop(id, d, label)?;
    s.active_mask = active;
    Ok(json!({"activeMask": active}))
}

/// Most pieces a tapered ⇧-click segment is cut into.
const MAX_TAPER_STEPS: usize = 64;

/// The strokes a ⇧-click adds: `to` drawn on from point `from`, the last point of stroke `last` (a straight segment
/// to `to`'s first point, then the rest of `to`). When the brush's size, feather or flow changed since `last`, the
/// segment is cut into pieces whose values step from `last`'s to `to`'s, so the line tapers; strokes combine by their
/// maximum, so the pieces join without building up.
fn connected_strokes(from: Point, last: &BrushStroke, to: BrushStroke) -> Vec<BrushStroke> {
    let Some(&first) = to.points.first() else { return vec![to] };
    let same = |a: f64, b: f64| (a - b).abs() <= 1e-9;
    // pen pressure along the line: the last dab's at its start, the click's at its end (none when neither has any)
    let pressured = !last.pressure.is_empty() || !to.pressure.is_empty();
    let (p_from, p_to) = (last.pressure_at(0), to.pressure_at(0));
    if same(last.size, to.size) && same(last.feather, to.feather) && same(last.flow, to.flow) {
        let mut points = Vec::with_capacity(to.points.len() + 1);
        points.push(from);
        points.extend(to.points.iter().copied());
        let pressure = if pressured { std::iter::once(p_from).chain((0..to.points.len()).map(|i| to.pressure_at(i))).collect() } else { Vec::new() };
        return vec![BrushStroke { points, pressure, ..to }];
    }
    // about 5% of size, or 2 points of feather or flow, per piece
    let ratio = if last.size > 0.0 && to.size > 0.0 { (to.size / last.size).ln().abs() / 1.05f64.ln() } else { 0.0 };
    let needed = [ratio, (to.feather - last.feather).abs() / 2.0, (to.flow - last.flow).abs() / 2.0]
        .into_iter()
        .filter(|v| v.is_finite())
        .fold(1.0f64, f64::max);
    let steps = (needed.ceil() as usize).clamp(1, MAX_TAPER_STEPS);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let at = |t: f64| Point::new(lerp(from.x, first.x, t), lerp(from.y, first.y, t));
    let mut out: Vec<BrushStroke> = (0..steps)
        .map(|i| {
            let (t0, t1) = (i as f64 / steps as f64, (i + 1) as f64 / steps as f64);
            let p = |t: f64| (f64::from(p_from) + (f64::from(p_to) - f64::from(p_from)) * t) as f32;
            BrushStroke {
                points: vec![at(t0), at(t1)],
                size: lerp(last.size, to.size, t1),
                feather: lerp(last.feather, to.feather, t1),
                flow: lerp(last.flow, to.flow, t1),
                pressure: if pressured { vec![p(t0), p(t1)] } else { Vec::new() },
                ..to.clone()
            }
        })
        .collect();
    if to.points.len() > 1 {
        out.push(to);
    }
    out
}

fn mask_id(p: &Value, active: Option<u32>, c: &str) -> Result<u32> {
    p.get("id").and_then(Value::as_u64).map(|v| v as u32).or(active).ok_or_else(|| bad(c, "no mask selected (give `id`)"))
}

fn find(masks: &mut [Mask], id: u32, c: &str) -> Result<usize> {
    masks.iter().position(|m| m.id == id).ok_or_else(|| bad(c, format!("no mask {id}")))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "mask.add",
            "Create New Mask",
            [],
            None,
            "{kind: brush|linear|radial|sky|subject|background|luminanceRange|colorRange|object|prompt, ...shape params (start/end, center/rx/ry/angle/feather, lo/hi…; object: points/exclude [[x,y],…]; prompt: text; object/prompt: seg? a stored segmentation, no model needed), name?} — AI kinds need the SAM 3 model; in the app a prompt returns {pending} and the mask appears when found",
            has_active,
            |s, p| {
                let kind = str_param(p, "kind").unwrap_or("radial").to_string();
                let mut shape = shape_from(&kind, p, "mask.add")?;
                let later = resolve_ai(s, &mut shape, "mask.add")?;
                let name = str_param(p, "name").map(str::to_string);
                if let Later::Text(text) = &later {
                    let id = s.active().ok_or_else(|| bad("mask.add", "no active photo"))?;
                    segment(s, |s| s.segment_text_later(id, None, "add", name, text))?;
                    return Ok(json!({"pending": true}));
                }
                let next = s.active().and_then(|id| s.develop_of(id)).map(|d| d.next_mask_id()).unwrap_or(1);
                // the effect last applied to a mask of this kind
                let adjust = super::mask_effects::new_mask_adjust(s, &kind);
                let out = masks_edit(s, "mask.add", "Add Mask", |masks, active| {
                    masks.push(Mask {
                        id: next,
                        name: name.unwrap_or_else(|| format!("Mask {next}")),
                        components: vec![MaskComponent { name: None, op: MaskOp::Add, invert: false, shape }],
                        adjust,
                        ..Default::default()
                    });
                    *active = Some(next);
                    Ok(())
                })?;
                queue_later(s, later, next, 0, out)
            }
        ),
        cmd!(
            "mask.addComponent",
            "Add/Subtract/Intersect Mask",
            [],
            None,
            "{id?: maskId, op: add|subtract|intersect, kind, ...shape params}",
            has_active,
            |s, p| {
                let kind = str_param(p, "kind").unwrap_or("brush").to_string();
                let op: MaskOp =
                    serde_json::from_value(p.get("op").cloned().unwrap_or(json!("add"))).map_err(|e| bad("mask.addComponent", e.to_string()))?;
                let active = s.active_mask;
                let mid = mask_id(p, active, "mask.addComponent")?;
                let id = s.active().ok_or_else(|| bad("mask.addComponent", "no active photo"))?;
                if !s.develop_of(id).is_some_and(|d| d.masks.iter().any(|m| m.id == mid)) {
                    return Err(bad("mask.addComponent", format!("no mask {mid}")));
                }
                let mut shape = shape_from(&kind, p, "mask.addComponent")?;
                let later = resolve_ai(s, &mut shape, "mask.addComponent")?;
                if let Later::Text(text) = &later {
                    let op = p.get("op").and_then(Value::as_str).unwrap_or("add");
                    segment(s, |s| s.segment_text_later(id, Some(mid), op, None, text))?;
                    return Ok(json!({"pending": true}));
                }
                let mut comp = 0;
                let out = masks_edit(s, "mask.addComponent", "Edit Mask", |masks, _| {
                    let i = find(masks, mid, "mask.addComponent")?;
                    masks[i].components.push(MaskComponent { name: None, op, invert: bool_or(p, "invert", false), shape });
                    comp = masks[i].components.len() - 1;
                    Ok(())
                })?;
                queue_later(s, later, mid, comp, out)
            }
        ),
        cmd!(
            "mask.brushStroke",
            "Brush Stroke",
            [],
            None,
            "{id?: maskId, points: [[x,y],…] normalized, size?: fraction of long edge (0.03), feather?, flow?, density?, erase?: bool, autoMask?: bool, pressure?: [p,…] pen pressure 0..1 per point (scales each dab's size and flow; omitted: full), connect?: bool — start from the last point of the mask's last stroke (⇧-click), size / feather / flow tapering from that stroke's to these} → {activeMask, connected}",
            has_active,
            |s, p| {
                let pts: Vec<Point> = p
                    .get("points")
                    .and_then(Value::as_array)
                    .ok_or_else(|| bad("mask.brushStroke", "missing points"))?
                    .iter()
                    .filter_map(|q| Some(Point::new(q.get(0)?.as_f64()?, q.get(1)?.as_f64()?)))
                    .collect();
                let pts_len = pts.len();
                let d = BrushStroke::default();
                let stroke = BrushStroke {
                    points: pts,
                    size: f64_or(p, "size", d.size),
                    feather: f64_or(p, "feather", d.feather),
                    flow: f64_or(p, "flow", d.flow),
                    density: f64_or(p, "density", d.density),
                    erase: bool_or(p, "erase", false),
                    auto_mask: bool_or(p, "autoMask", false),
                    pressure: pressure_list(p.get("pressure"), pts_len)?,
                };
                let connect = bool_or(p, "connect", false);
                let mut connected = false;
                let next = s.active().and_then(|id| s.develop_of(id)).map(|d| d.next_mask_id()).unwrap_or(1);
                let want = p.get("id").and_then(Value::as_u64).map(|v| v as u32).or(s.active_mask);
                let adjust = super::mask_effects::new_mask_adjust(s, "brush");
                let mut out = masks_edit(s, "mask.brushStroke", "Brush", |masks, active| {
                    let i = match want.and_then(|m| masks.iter().position(|x| x.id == m)) {
                        Some(i) => i,
                        None => {
                            masks.push(Mask { id: next, name: format!("Mask {next}"), adjust, ..Default::default() });
                            *active = Some(next);
                            masks.len() - 1
                        }
                    };
                    let m = &mut masks[i];
                    if let Some(MaskComponent { shape: MaskShape::Brush { strokes }, .. }) =
                        m.components.iter_mut().rev().find(|c| matches!(c.shape, MaskShape::Brush { .. }))
                    {
                        // the previous stroke's end and its brush values (not its points)
                        let anchor = strokes.last().filter(|_| connect).and_then(|l| {
                            Some((
                                *l.points.last()?,
                                BrushStroke {
                                    points: Vec::new(),
                                    size: l.size,
                                    feather: l.feather,
                                    flow: l.flow,
                                    // the pressure the last dab was painted with, when the stroke recorded any
                                    pressure: if l.pressure.is_empty() { Vec::new() } else { vec![l.pressure_at(l.points.len().saturating_sub(1))] },
                                    ..BrushStroke::default()
                                },
                            ))
                        });
                        match anchor {
                            Some((from, last)) => {
                                strokes.extend(connected_strokes(from, &last, stroke));
                                connected = true;
                            }
                            None => strokes.push(stroke),
                        }
                    } else {
                        m.components.push(MaskComponent {
                            name: None,
                            op: MaskOp::Add,
                            invert: false,
                            shape: MaskShape::Brush { strokes: vec![stroke] },
                        });
                    }
                    Ok(())
                })?;
                if let Some(o) = out.as_object_mut() {
                    o.insert("connected".into(), json!(connected));
                }
                Ok(out)
            }
        ),
        cmd!("mask.update", "Update Mask Shape", [], None, "{id?, component?: index (0), shape: MaskShape JSON}", has_active, |s, p| {
            let shape: MaskShape =
                serde_json::from_value(p.get("shape").cloned().unwrap_or_default()).map_err(|e| bad("mask.update", e.to_string()))?;
            let comp = p.get("component").and_then(Value::as_u64).unwrap_or(0) as usize;
            let mid = mask_id(p, s.active_mask, "mask.update")?;
            masks_edit(s, "mask.update", "Edit Mask", |masks, _| {
                let i = find(masks, mid, "mask.update")?;
                let c = masks[i].components.get_mut(comp).ok_or_else(|| bad("mask.update", "no such component"))?;
                c.shape = shape;
                Ok(())
            })
        }),
        cmd!(
            "mask.sampleColor",
            "Sample Mask Color",
            [],
            None,
            "{x, y: normalized image point, add?: bool (⇧: add to the samples, max 5), id?, component?} — the colour range of the selected mask (its first colour-range component) samples the colour there → {samples}",
            has_active,
            |s, p| {
                let c = "mask.sampleColor";
                let (x, y) = (super::f64_req(p, "x", c)?, super::f64_req(p, "y", c)?);
                let add = p.get("add").and_then(Value::as_bool).unwrap_or(false);
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                let src = s.source_now(id, crate::media::SourceLevel::Thumb).map_err(|e| bad(c, e))?;
                let info = s.source_info(id);
                let d = s.develop_of(id).unwrap_or_default();
                let req = lightcraft_pipeline::RenderRequest::fit(384, 384);
                let lab = lightcraft_pipeline::color_range_sample(&src, &info, &d, &req, lightcraft_geom::Point::new(x, y))
                    .ok_or_else(|| bad(c, "the point is outside the photo"))?;
                let mid = mask_id(p, s.active_mask, c)?;
                let comp = p.get("component").and_then(Value::as_u64).map(|v| v as usize);
                let mut out = Vec::new();
                masks_edit(s, c, "Color Range", |masks, _| {
                    let i = find(masks, mid, c)?;
                    let k = comp
                        .or_else(|| masks[i].components.iter().position(|x| matches!(x.shape, MaskShape::ColorRange { .. })))
                        .ok_or_else(|| bad(c, "the mask has no colour range"))?;
                    let Some(MaskShape::ColorRange { samples, .. }) = masks[i].components.get_mut(k).map(|x| &mut x.shape) else {
                        return Err(bad(c, "not a colour range component"));
                    };
                    if !add {
                        samples.clear();
                    }
                    if samples.len() < 5 {
                        samples.push(lab);
                    }
                    out = samples.clone();
                    Ok(())
                })?;
                Ok(json!({"samples": out}))
            }
        ),
        cmd!(
            "mask.objectPoint",
            "Add Object Click",
            [],
            None,
            "{x, y: normalized image point, exclude?: bool (⌥-click: leave this part out), id?: maskId} — a click on the selected mask's Object selection (its last Object component); SAM 3 re-segments the object → {include, exclude}",
            has_active,
            |s, p| {
                let c = "mask.objectPoint";
                let at = Point::new(super::f64_req(p, "x", c)?, super::f64_req(p, "y", c)?);
                if !(0.0..=1.0).contains(&at.x) || !(0.0..=1.0).contains(&at.y) {
                    return Err(bad(c, "the point is outside the photo"));
                }
                let exclude = bool_or(p, "exclude", false);
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                let mid = mask_id(p, s.active_mask, c)?;
                let d = s.develop_of(id).unwrap_or_default();
                let m = d.masks.iter().find(|m| m.id == mid).ok_or_else(|| bad(c, format!("no mask {mid}")))?;
                let k = m
                    .components
                    .iter()
                    .rposition(|x| matches!(x.shape, MaskShape::Object { .. }))
                    .ok_or_else(|| bad(c, "the mask has no Object selection (add one first)"))?;
                let Some(MaskShape::Object { hint, exclude: ex, edge, .. }) = m.components.get(k).map(|x| &x.shape) else {
                    return Err(bad(c, "not an Object component"));
                };
                let (mut hint, mut ex, edge) = (hint.clone(), ex.clone(), *edge);
                // clicks still on their way to the model count too
                if let Some(pending) = s.segmenter.pending_clicks().filter(|q| (q.photo, q.mask, q.comp) == (id, mid, k)) {
                    (hint, ex) = (pending.hint.clone(), pending.exclude.clone());
                }
                if hint.len() + ex.len() >= MAX_CLICKS {
                    return Err(ai_err(format!("an Object selection takes at most {MAX_CLICKS} clicks; start a new one")));
                }
                if exclude {
                    ex.push(at)
                } else {
                    hint.push(at)
                }
                if s.segmenter.background && !hint.is_empty() {
                    let out = json!({"include": hint.len(), "exclude": ex.len(), "pending": true});
                    segment(s, |s| s.segment_clicks_later(id, mid, k, hint, ex))?;
                    return Ok(out);
                }
                let seg = if hint.is_empty() { None } else { Some(segment(s, |s| s.segment_clicks(id, &clicks_of(&hint, &ex)))?) };
                let out = json!({"include": hint.len(), "exclude": ex.len()});
                masks_edit(s, c, "Object Mask", |masks, _| {
                    let i = find(masks, mid, c)?;
                    if let Some(comp) = masks[i].components.get_mut(k) {
                        comp.shape = MaskShape::Object { hint, exclude: ex, seg, detail: vec![], edge };
                    }
                    Ok(())
                })?;
                Ok(out)
            }
        ),
        cmd!(
            "mask.refineDetail",
            "Refine AI Mask Detail",
            [],
            None,
            "{id?: maskId, component?} — a zoomed-in SAM 3 pass over an Object/Describe selection (default: the mask's last one): the photo around it is analyzed again at a higher resolution, in the background; the mask updates when it's done → {started}",
            has_active,
            |s, p| {
                let c = "mask.refineDetail";
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                let mid = mask_id(p, s.active_mask, c)?;
                let d = s.develop_of(id).unwrap_or_default();
                let m = d.masks.iter().find(|m| m.id == mid).ok_or_else(|| bad(c, format!("no mask {mid}")))?;
                let k = match p.get("component").and_then(Value::as_u64) {
                    Some(k) => k as usize,
                    None => m
                        .components
                        .iter()
                        .rposition(|x| matches!(x.shape, MaskShape::Object { .. } | MaskShape::Prompt { .. }))
                        .ok_or_else(|| bad(c, "the mask has no Object or Describe selection"))?,
                };
                let started = segment(s, |s| s.segment_detail(id, mid, k))?;
                Ok(json!({"started": started}))
            }
        ),
        cmd!(
            query "segment.prepare",
            "Prepare AI Masks",
            [],
            None,
            "{} — load SAM 3 and analyze the active photo in the background, so Object clicks are instant → {busy}",
            has_active,
            |s, _| {
                let c = "segment.prepare";
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                segment(s, |s| s.segment_prepare(id))?;
                Ok(json!({"busy": s.segmenter.busy()}))
            }
        ),
        cmd!(
            query "segment.model.status",
            "AI Mask Model Status",
            [],
            None,
            "{} — whether this build has AI masks, whether the SAM 3 model is installed (and where), loaded, busy, and the download's progress → {available, installed, dir, loaded, busy, analyzing, sizeBytes, license, licenseUrl, mirrors, download: {running, done, total, file, error, finished}}",
            super::always,
            |s, _| {
                let g = &s.segmenter;
                Ok(json!({
                    "available": crate::segment::Segmenter::AVAILABLE,
                    "installed": g.installed(),
                    "dir": g.dir.as_ref().map(|d| d.display().to_string()),
                    "remote": g.remote_endpoint(),
                    "loaded": g.loaded(),
                    "busy": g.busy(),
                    "analyzing": g.analyzing(),
                    "sizeBytes": crate::segment::MODEL_BYTES,
                    "license": crate::segment::LICENSE_NAME,
                    "licenseUrl": crate::segment::LICENSE_URL,
                    "mirrors": g.mirrors().len(),
                    "download": g.download_status(),
                }))
            }
        ),
        cmd!(
            query "segment.model.download",
            "Download AI Mask Model",
            [],
            None,
            "{acknowledged: true} — download the SAM 3 model (about 3.4 GB, Meta's SAM License, not LightCraft's) in the background, from the configured mirrors; only after the user agreed to it. Watch segment.model.status; segment.model.cancel stops it (it resumes later) → {started, installed, downloading}",
            super::always,
            |s, p| {
                let c = "segment.model.download";
                if p.get("acknowledged").and_then(Value::as_bool) != Some(true) {
                    return Err(bad(
                        c,
                        format!(
                            "the SAM 3 model is a {:.1} GB download under Meta's {} ({}): pass `acknowledged: true` once the user has agreed to download it",
                            crate::segment::MODEL_BYTES as f64 / 1e9,
                            crate::segment::LICENSE_NAME,
                            crate::segment::LICENSE_URL
                        ),
                    ));
                }
                let started = s.segmenter.start_download(&s.activity).map_err(ai_err)?;
                Ok(json!({"started": started, "installed": s.segmenter.installed(), "downloading": s.segmenter.download_status().running}))
            }
        ),
        cmd!(
            query "segment.model.cancel",
            "Cancel AI Mask Model Download",
            [],
            None,
            "{} — stop the SAM 3 download (what has arrived is kept, and a new download resumes from it) → {cancelled}",
            super::always,
            |s, _| Ok(json!({"cancelled": s.segmenter.cancel_download()}))
        ),
        cmd!(
            "mask.refine",
            "Refine Mask Edges",
            [],
            None,
            "{id?, value: 0..100} — the mask's edges snap to the photo's (guided filter; rendered on the CPU)",
            has_active,
            |s, p| {
                let v = f64_or(p, "value", 0.0).clamp(0.0, 100.0);
                let mid = mask_id(p, s.active_mask, "mask.refine")?;
                masks_edit(s, "mask.refine", "Refine Edges", |masks, _| {
                    let i = find(masks, mid, "mask.refine")?;
                    masks[i].refine = v;
                    Ok(())
                })
            }
        ),
        cmd!(
            "mask.adjust",
            "Set Mask Adjustments",
            [],
            None,
            "{id?, values: {exposure, contrast, highlights, shadows, whites, blacks, temp, tint, texture, clarity, dehaze, hue, saturation, sharpness, noise, moire, defringe, color_hue, color_sat, amount}}",
            has_active,
            |s, p| {
                let vals = p.get("values").cloned().ok_or_else(|| bad("mask.adjust", "missing values"))?;
                let mid = mask_id(p, s.active_mask, "mask.adjust")?;
                masks_edit(s, "mask.adjust", "Mask Adjustment", |masks, _| {
                    let i = find(masks, mid, "mask.adjust")?;
                    let mut v = serde_json::to_value(masks[i].adjust).unwrap_or_default();
                    lightcraft_develop::presets::deep_merge(&mut v, &vals);
                    let a: LocalAdjustments = serde_json::from_value(v).map_err(|e| bad("mask.adjust", e.to_string()))?;
                    masks[i].adjust = clamp_local(a);
                    Ok(())
                })
            }
        ),
        cmd!("mask.select", "Select Mask", [], None, "{id|null}", has_active, |s, p| {
            s.active_mask = p.get("id").and_then(Value::as_u64).map(|v| v as u32);
            ok()
        }),
        cmd!("mask.delete", "Delete Mask", [], None, "{id?}", has_active, |s, p| {
            let mid = mask_id(p, s.active_mask, "mask.delete")?;
            masks_edit(s, "mask.delete", "Delete Mask", |masks, active| {
                let i = find(masks, mid, "mask.delete")?;
                masks.remove(i);
                *active = masks.last().map(|m| m.id);
                Ok(())
            })
        }),
        cmd!("mask.deleteAll", "Delete All Masks", [], None, "{}", has_active, |s, _| {
            masks_edit(s, "mask.deleteAll", "Delete All Masks", |masks, active| {
                masks.clear();
                *active = None;
                Ok(())
            })
        }),
        cmd!("mask.rename", "Rename Mask", [], None, "{id?, name}", has_active, |s, p| {
            let name = str_param(p, "name").ok_or_else(|| bad("mask.rename", "missing name"))?.to_string();
            let mid = mask_id(p, s.active_mask, "mask.rename")?;
            masks_edit(s, "mask.rename", "Rename Mask", |masks, _| {
                let i = find(masks, mid, "mask.rename")?;
                masks[i].name = name;
                Ok(())
            })
        }),
        cmd!("mask.invert", "Invert Mask", [], None, "{id?}", has_active, |s, p| {
            let mid = mask_id(p, s.active_mask, "mask.invert")?;
            masks_edit(s, "mask.invert", "Invert Mask", |masks, _| {
                let i = find(masks, mid, "mask.invert")?;
                masks[i].invert = !masks[i].invert;
                Ok(())
            })
        }),
        cmd!("mask.visible", "Show/Hide Mask", [], None, "{id?, visible?: bool}", has_active, |s, p| {
            let mid = mask_id(p, s.active_mask, "mask.visible")?;
            masks_edit(s, "mask.visible", "Toggle Mask", |masks, _| {
                let i = find(masks, mid, "mask.visible")?;
                masks[i].visible = bool_or(p, "visible", !masks[i].visible);
                Ok(())
            })
        }),
        cmd!(
            "mask.duplicate",
            "Duplicate Mask",
            [],
            None,
            "{id?, invert?: bool (Duplicate and Invert)} — the copy goes right after the original and is selected",
            has_active,
            |s, p| {
                let mid = mask_id(p, s.active_mask, "mask.duplicate")?;
                let invert = bool_or(p, "invert", false);
                let next = s.active().and_then(|id| s.develop_of(id)).map(|d| d.next_mask_id()).unwrap_or(1);
                let label = if invert { "Duplicate and Invert Mask" } else { "Duplicate Mask" };
                masks_edit(s, "mask.duplicate", label, |masks, active| {
                    let i = find(masks, mid, "mask.duplicate")?;
                    let mut m = masks[i].clone();
                    m.id = next;
                    m.name = format!("{} copy", m.name);
                    if invert {
                        m.invert = !m.invert;
                    }
                    masks.insert(i + 1, m);
                    *active = Some(next);
                    Ok(())
                })
            }
        ),
        cmd!(
            "mask.component",
            "Edit Mask Component",
            [],
            None,
            "{id?, component: index, action: invert|duplicate|delete|rename|op, name? (rename; empty clears), op?: add|subtract|intersect} — one component of a mask (deleting the last one deletes the mask)",
            has_active,
            |s, p| {
                let c = "mask.component";
                let mid = mask_id(p, s.active_mask, c)?;
                let k = p.get("component").and_then(Value::as_u64).ok_or_else(|| bad(c, "missing component"))? as usize;
                let action = str_param(p, "action").ok_or_else(|| bad(c, "missing action"))?.to_string();
                let op: Option<MaskOp> = match p.get("op") {
                    Some(v) => Some(serde_json::from_value(v.clone()).map_err(|e| bad(c, e.to_string()))?),
                    None => None,
                };
                let label = match action.as_str() {
                    "invert" => "Invert Component",
                    "duplicate" => "Duplicate Component",
                    "delete" => "Delete Component",
                    "rename" => "Rename Component",
                    "op" => "Change Component Mode",
                    a => return Err(bad(c, format!("unknown action {a:?} (invert|duplicate|delete|rename|op)"))),
                };
                masks_edit(s, c, label, |masks, active| {
                    let i = find(masks, mid, c)?;
                    let comps = &mut masks[i].components;
                    let comp = comps.get_mut(k).ok_or_else(|| bad(c, "no such component"))?;
                    match action.as_str() {
                        "invert" => comp.invert = !comp.invert,
                        "rename" => comp.name = str_param(p, "name").map(str::trim).filter(|n| !n.is_empty()).map(str::to_string),
                        "op" => comp.op = op.ok_or_else(|| bad(c, "missing op"))?,
                        "duplicate" => {
                            let copy = comp.clone();
                            comps.insert(k + 1, copy);
                        }
                        _ => {
                            comps.remove(k);
                            if comps.is_empty() {
                                masks.remove(i);
                                *active = masks.last().map(|m| m.id);
                            }
                        }
                    }
                    Ok(())
                })
            }
        ),
        cmd!("mask.move", "Move Mask", [], None, "{id?, to?: index (0 = top), delta?: ±n} — reorder the masks list", has_active, |s, p| {
            let mid = mask_id(p, s.active_mask, "mask.move")?;
            let mut masks = s.active().and_then(|id| s.develop_of(id)).map(|d| d.masks.clone()).unwrap_or_default();
            let i = find(&mut masks, mid, "mask.move")?;
            let to = match (p.get("to").and_then(Value::as_i64), p.get("delta").and_then(Value::as_i64)) {
                (Some(t), _) => t,
                // (an agent's delta can be anything: saturate rather than overflow)
                (None, Some(d)) => (i as i64).saturating_add(d),
                (None, None) => return Err(bad("mask.move", "give `to` (index) or `delta`")),
            }
            .clamp(0, masks.len() as i64 - 1) as usize;
            if to == i {
                return Ok(json!({"activeMask": s.active_mask}));
            }
            masks_edit(s, "mask.move", "Reorder Masks", |masks, _| {
                let i = find(masks, mid, "mask.move")?;
                let m = masks.remove(i);
                masks.insert(to, m);
                Ok(())
            })
        }),
        // ---- Remove tool (spots)
        cmd!(
            "spot.findDust",
            "Find Dust Spots",
            [],
            None,
            "{sensitivity?: 0..100 (50), add?: bool (true)} — find sensor-dust spots (small soft dark spots on smooth areas) and add a heal spot on each (one undo step) → {spots: [{x, y, size}], added}",
            has_active,
            |s, p| {
                let c = "spot.findDust";
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                // the uncropped photo, so positions are the spots' own coordinates
                let job = s.render_job(id, 1600, 1600, false, false).ok_or_else(|| bad(c, "no photo"))?;
                let img = job.run().rendered.map_err(|e| bad(c, e))?.image;
                let found = lightcraft_pipeline::dust::detect(&img, f64_or(p, "sensitivity", 50.0) as f32);
                let out: Vec<Value> = found.iter().map(|d| json!({"x": d.x, "y": d.y, "size": d.radius})).collect();
                if !p.get("add").and_then(Value::as_bool).unwrap_or(true) || found.is_empty() {
                    return Ok(json!({"spots": out, "added": 0}));
                }
                let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
                let base = Spot::default();
                for d in &found {
                    let mut spot = Spot {
                        mode: SpotMode::Heal,
                        points: vec![Point::new(d.x, d.y)],
                        size: d.radius.clamp(SPOT_SIZE.0, SPOT_SIZE.1),
                        feather: base.feather,
                        opacity: 100.0,
                        source_offset: None,
                    };
                    spot.source_offset = pick_source(s, id, &dd, &spot, None);
                    dd.spots.push(spot);
                }
                s.set_develop(id, dd, "Find Dust Spots")?;
                Ok(json!({"spots": out, "added": found.len()}))
            }
        ),
        cmd!(
            "spot.findBlemishes",
            "Find Blemishes",
            [],
            None,
            "{sensitivity?: 0..100 (50), add?: bool (true), maxPerFace?: 1..20 (6), allSelected?: bool (false)} — find small blemishes (darker or redder round spots on skin) on the faces of the active photo (or all selected photos when `allSelected: true`) and add a heal spot on each (one undo step) → {faces, spots: [{x, y, size, face}], added, photos}",
            has_active,
            |s, p| {
                let c = "spot.findBlemishes";
                let active_id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                let all_selected = bool_or(p, "allSelected", false);
                let target_ids: Vec<crate::PhotoId> = if all_selected && !s.selection.ids.is_empty() {
                    s.selection.ids.clone()
                } else {
                    vec![active_id]
                };
                let max_per_face = (f64_or(p, "maxPerFace", 6.0) as usize).clamp(1, 20);
                let add_spots = p.get("add").and_then(Value::as_bool).unwrap_or(true);

                let mut total_faces = 0usize;
                let mut total_added = 0usize;
                let mut active_out: Vec<Value> = Vec::new();
                let mut ops: Vec<lightcraft_catalog::Op> = Vec::new();
                let mut touched_photos = 0usize;

                for id in target_ids {
                    // the uncropped photo, so positions are the spots' own coordinates
                    let Some(job) = s.render_job(id, 1600, 1600, false, false) else {
                        continue;
                    };
                    let Ok(rendered) = job.run().rendered else {
                        continue;
                    };
                    let img = rendered.image;
                    let detector_opt = super::face_detect::detector(s).ok().or_else(|| lightcraft_faces::yunet::builtin().cloned());
                    let mut boxes: Vec<lightcraft_pipeline::blemish::FaceBox> = match detector_opt {
                        Some(detector) => {
                            let rgb: Vec<u8> = img.data.iter().flat_map(|px| [px[0], px[1], px[2]]).collect();
                            match detector.detect(&rgb, img.width, img.height, &lightcraft_faces::yunet::Options::default()) {
                                Ok(faces) => faces
                                    .iter()
                                    .map(|f| lightcraft_pipeline::blemish::FaceBox {
                                        x0: f64::from(f.x0),
                                        y0: f64::from(f.y0),
                                        x1: f64::from(f.x1),
                                        y1: f64::from(f.y1),
                                        landmarks: f.landmarks.map(|(x, y)| (f64::from(x), f64::from(y))),
                                    })
                                    .collect(),
                                Err(_) => Vec::new(),
                            }
                        }
                        None => Vec::new(),
                    };
                    if boxes.is_empty()
                        && let Some(ph) = s.catalog.photo(id)
                    {
                        boxes = ph
                            .meta
                            .regions
                            .iter()
                            .filter(|r| r.kind == lightcraft_meta::RegionKind::Face)
                            .map(|r| lightcraft_pipeline::blemish::face_from_rect(r.rect.x0, r.rect.y0, r.rect.x1, r.rect.y1))
                            .collect();
                    }
                    if boxes.is_empty() {
                        boxes = lightcraft_pipeline::blemish::detect_skin_faces(&img);
                    }
                    total_faces += boxes.len();
                    let default_sens = s.develop_of(id).map(|d| d.retouch.blemish_sensitivity).unwrap_or(50.0);
                    let mut found = lightcraft_pipeline::blemish::detect(&img, &boxes, f64_or(p, "sensitivity", default_sens) as f32);
                    let mut per_face = std::collections::BTreeMap::new();
                    found.retain(|b| {
                        let n = per_face.entry(b.face).or_insert(0usize);
                        *n += 1;
                        *n <= max_per_face
                    });
                    if id == active_id {
                        active_out = found.iter().map(|b| json!({"x": b.x, "y": b.y, "size": b.radius, "face": b.face})).collect();
                    }
                    if !add_spots || found.is_empty() {
                        continue;
                    }
                    let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
                    let base = Spot::default();
                    for b in &found {
                        let mut spot = Spot {
                            mode: SpotMode::Heal,
                            points: vec![Point::new(b.x, b.y)],
                            size: b.radius.clamp(SPOT_SIZE.0, SPOT_SIZE.1),
                            feather: base.feather,
                            opacity: 100.0,
                            source_offset: None,
                        };
                        spot.source_offset = pick_source(s, id, &dd, &spot, None);
                        dd.spots.push(spot);
                    }
                    total_added += found.len();
                    if let Some(op) = s.develop_op(id, dd, "Find Blemishes") {
                        ops.push(op);
                        touched_photos += 1;
                    }
                }

                if !ops.is_empty() {
                    s.commit("Find Blemishes", lightcraft_catalog::Op::Batch { ops })?;
                }
                Ok(json!({"faces": total_faces, "spots": active_out, "added": total_added, "photos": touched_photos}))
            }
        ),
        cmd!(
            query "retouch.analyze",
            "Analyze Portrait for AI Retouch",
            [],
            None,
            "{id?} — run YuNet ONNX neural 5-landmark detection and Oklab frequency-separation skin/feature analysis on the photo → {id, fileName, imageDataUrl, meta, analysis, current}",
            super::always,
            |s, p| {
                let id_opt = p
                    .get("id")
                    .and_then(Value::as_u64)
                    .map(crate::PhotoId)
                    .or_else(|| s.active())
                    .or_else(|| s.catalog.photos().next().map(|ph| ph.id));
                let Some(id) = id_opt else {
                    return Ok(json!({
                        "id": null,
                        "fileName": "",
                        "imageDataUrl": "",
                        "meta": {},
                        "analysis": null,
                        "current": {}
                    }));
                };

                let rendered_opt = s
                    .render_job(id, 384, 384, false, true)
                    .and_then(|job| job.run().rendered.ok())
                    .or_else(|| s.thumb_job(id, 384).and_then(|job| job.run().rendered.ok()));

                let (image_data_url, analysis) = if let Some(rendered) = rendered_opt {
                    let rgba = rendered.image;
                    let enc_img = lightcraft_codecs::EncodeImage::rgba8(&rgba);
                    let jpeg_bytes = lightcraft_codecs::encode_jpeg(
                        &enc_img,
                        78,
                        lightcraft_codecs::ChromaSubsampling::S420,
                        &lightcraft_codecs::EncodeMeta::default(),
                    )
                    .unwrap_or_default();
                    let data_url = if jpeg_bytes.is_empty() {
                        String::new()
                    } else {
                        format!("data:image/jpeg;base64,{}", encode_base64(&jpeg_bytes))
                    };
                    let mut lin = lightcraft_raster::Rgb32f::new(rgba.width, rgba.height);
                    for (dst, src) in lin.data.iter_mut().zip(rgba.data.iter()) {
                        let r = (f32::from(src[0]) / 255.0).powf(2.2);
                        let g = (f32::from(src[1]) / 255.0).powf(2.2);
                        let b = (f32::from(src[2]) / 255.0).powf(2.2);
                        *dst = [r, g, b];
                    }
                    (data_url, lightcraft_pipeline::retouch::analyze_portrait(&lin))
                } else {
                    (String::new(), lightcraft_pipeline::retouch::analyze_portrait(&lightcraft_raster::Rgb32f::new(0, 0)))
                };

                let cur = s.develop_of(id).unwrap_or_default();
                let k_to_rel = |k: f64| -> f64 {
                    ((1e6 / 6500.0 - 1e6 / k.clamp(2000.0, 50000.0)) / 0.8).clamp(-100.0, 100.0)
                };
                let (file_name, meta_json) = if let Some(ph) = s.catalog.photo(id) {
                    (
                        ph.file_name.clone(),
                        json!({
                            "width": ph.width,
                            "height": ph.height,
                            "camera": ph.meta.camera,
                            "lens": ph.meta.lens,
                            "iso": ph.meta.iso,
                            "aperture": ph.meta.aperture,
                            "shutter": ph.meta.shutter,
                            "focalMm": ph.meta.focal_mm,
                            "rating": ph.rating,
                            "masksCount": cur.masks.len(),
                            "spotsCount": cur.spots.len(),
                        }),
                    )
                } else {
                    (String::new(), json!({}))
                };
                Ok(json!({
                    "id": id.0,
                    "fileName": file_name,
                    "imageDataUrl": image_data_url,
                    "meta": meta_json,
                    "analysis": analysis,
                    "current": {
                        "retouch": cur.retouch,
                        "light": cur.light,
                        "wb": {
                            "temp": k_to_rel(cur.wb.temp).round(),
                            "kelvin": cur.wb.temp.round(),
                            "tint": cur.wb.tint,
                        },
                        "color": cur.color,
                        "grading": cur.grading,
                        "mixer": cur.mixer,
                        "effects": {
                            "texture": cur.effects.texture,
                            "clarity": cur.effects.clarity,
                            "dehaze": cur.effects.dehaze,
                            "vignette": cur.vignette.amount,
                            "grain": cur.grain.amount,
                        },
                        "detail": {
                            "sharpness": cur.detail.sharpen_amount,
                            "nrLuminance": cur.detail.nr_luminance,
                            "nrColor": cur.detail.nr_color,
                        },
                        "treatment": cur.treatment,
                    }
                }))
            }
        ),
        cmd!(
            "retouch.aiApply",
            "AI Retouch Director",
            ["Develop"],
            None,
            "{response?: string, prompt?: string, allSelected?: bool} — apply AI Studio Assistant recipe from Gemini/OpenAI JSON completion or YuNet ONNX + Oklab image analysis → {changed, summary, diff, neural}",
            super::always,
            |s, p| {
                let all_selected = p.get("allSelected").and_then(Value::as_bool).unwrap_or(false);
                let prompt = p.get("prompt").and_then(Value::as_str).unwrap_or("").trim();
                let response_str = p.get("response").and_then(Value::as_str).unwrap_or("").trim();

                // Extract optional JSON object from LLM completion text (handles ```json ... ``` fences or raw JSON).
                let parsed_ai: Option<Value> = if !response_str.is_empty() {
                    let start = response_str.find('{');
                    let end = response_str.rfind('}');
                    if let (Some(i), Some(j)) = (start, end)
                        && j > i
                    {
                        serde_json::from_str::<Value>(&response_str[i..=j]).ok()
                    } else {
                        None
                    }
                } else {
                    None
                };

                let ai_reply: Option<String> = parsed_ai
                    .as_ref()
                    .and_then(|ai| {
                        ai.get("reply")
                            .or_else(|| ai.get("summary"))
                            .or_else(|| ai.get("message"))
                            .and_then(Value::as_str)
                    })
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .or_else(|| {
                        if !response_str.is_empty() && parsed_ai.is_none() {
                            Some(response_str.to_string())
                        } else {
                            None
                        }
                    });

                // Pure conversational / Q&A turn (no edits requested):
                let is_chat_only = parsed_ai.as_ref().is_some_and(|ai| {
                    ai.get("mode").and_then(Value::as_str).is_some_and(|m| m.eq_ignore_ascii_case("chat"))
                        || ai.get("isEdit").and_then(Value::as_bool) == Some(false)
                });
                if is_chat_only {
                    return Ok(json!({
                        "changed": 0,
                        "summary": ai_reply.unwrap_or_else(|| "Here to help — ask me anything about this photo or tell me what to adjust.".to_string()),
                        "neural": false,
                    }));
                }

                let active_opt = s.active().or_else(|| s.catalog.photos().next().map(|ph| ph.id));
                let Some(active_id) = active_opt else {
                    return Ok(json!({
                        "changed": 0,
                        "summary": ai_reply.unwrap_or_else(|| "Import or select a photo first so I can inspect and edit it.".to_string()),
                        "neural": false,
                    }));
                };

                let target_ids: Vec<crate::PhotoId> = if all_selected && !s.selection.ids.is_empty() {
                    s.selection.ids.clone()
                } else {
                    vec![active_id]
                };

                let rel_to_k = |rel: f64| -> f64 {
                    let r = rel.clamp(-100.0, 100.0);
                    (1e6 / (1e6 / 6500.0 - r * 0.8)).clamp(2000.0, 50000.0)
                };
                let k_to_rel = |k: f64| -> f64 {
                    ((1e6 / 6500.0 - 1e6 / k.clamp(2000.0, 50000.0)) / 0.8).clamp(-100.0, 100.0)
                };

                let actions: Vec<String> = parsed_ai
                    .as_ref()
                    .and_then(|ai| ai.get("actions"))
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .map(|s| s.trim().to_ascii_lowercase())
                            .collect()
                    })
                    .unwrap_or_default();

                let mut ops: Vec<lightcraft_catalog::Op> = Vec::new();
                let mut summary = ai_reply.unwrap_or_default();
                let mut any_neural = false;
                let mut diff_notes: Vec<String> = Vec::new();

                for id in target_ids {
                    let prev_dd = (*s.develop_of(id).unwrap_or_default()).clone();
                    let mut dd = prev_dd.clone();
                    let mut r = dd.retouch.clone();
                    let q = prompt.to_ascii_lowercase();

                    let want_auto_retouch = q.is_empty()
                        || actions.iter().any(|a| a == "autoretouch" || a == "auto_retouch")
                        || (parsed_ai.is_none()
                            && !dd.retouch.is_active()
                            && (q.contains("auto retouch")
                                || q.contains("auto-retouch")
                                || q == "retouch"
                                || q == "retouch portrait"));

                    if want_auto_retouch {
                        let base_settings = lightcraft_develop::DevelopSettings::default();
                        let analysis = if let Some(job) = s.preview_job(id, 512, 512, false, &base_settings)
                            && let Ok(rendered) = job.run().rendered
                        {
                            let rgba = rendered.image;
                            let mut lin = lightcraft_raster::Rgb32f::new(rgba.width, rgba.height);
                            for (dst, src) in lin.data.iter_mut().zip(rgba.data.iter()) {
                                let rv = (f32::from(src[0]) / 255.0).powf(2.2);
                                let gv = (f32::from(src[1]) / 255.0).powf(2.2);
                                let bv = (f32::from(src[2]) / 255.0).powf(2.2);
                                *dst = [rv, gv, bv];
                            }
                            lightcraft_pipeline::retouch::analyze_portrait(&lin)
                        } else {
                            lightcraft_pipeline::retouch::analyze_portrait(&lightcraft_raster::Rgb32f::new(0, 0))
                        };
                        if analysis.neural_detector {
                            any_neural = true;
                        }
                        r = analysis.recommended;
                    }

                    // Only run local keyword heuristics when no structured AI JSON response was provided:
                    if parsed_ai.is_none() && !q.is_empty() {
                        if q.contains("reset") || q.contains("original") || q.contains("clear all") {
                            dd = lightcraft_develop::DevelopSettings::default();
                            r = lightcraft_develop::Retouch::default();
                            summary = "Reset all image adjustments to original.".to_string();
                        }
                        if q.contains("natural") || q.contains("subtle") || q.contains("light retouch") {
                            r.skin_smooth = (r.skin_smooth.max(32.0) * 0.72).clamp(20.0, 50.0).round();
                            r.skin_texture = 68.0;
                            r.face_slim = 0.0;
                            r.eye_enlarge = 0.0;
                        }
                        if q.contains("glamour") || q.contains("beauty") || q.contains("flawless") || q.contains("smooth") {
                            r.skin_smooth = (r.skin_smooth + 22.0).clamp(55.0, 88.0);
                            r.skin_tone_even = (r.skin_tone_even + 18.0).clamp(50.0, 85.0);
                            r.skin_radiance = (r.skin_radiance + 14.0).clamp(20.0, 55.0);
                        }
                        if q.contains("editorial") || q.contains("magazine") || q.contains("pore") || q.contains("texture") {
                            r.skin_texture = 74.0;
                            r.face_contour = (r.face_contour + 20.0).clamp(38.0, 75.0);
                        }
                        if q.contains("male") || q.contains("men") || q.contains("masculine") {
                            r.skin_smooth = r.skin_smooth.clamp(25.0, 42.0);
                            r.skin_texture = 78.0;
                            r.lip_enhance = 0.0;
                            r.lip_moisture = 0.0;
                            r.eye_enlarge = 0.0;
                        }
                        if q.contains("shine") || q.contains("oil") || q.contains("sweat") || q.contains("matte") {
                            r.shine_control = (r.shine_control + 28.0).clamp(60.0, 95.0);
                        }
                        if q.contains("dark circle") || q.contains("eye bag") || q.contains("tired") || q.contains("undereye") || q.contains("under-eye") {
                            r.dark_circles = (r.dark_circles + 26.0).clamp(55.0, 92.0);
                        }
                        if q.contains("wrinkle") || q.contains("line") || q.contains("young") || q.contains("age") {
                            r.wrinkle_reduce = (r.wrinkle_reduce + 28.0).clamp(45.0, 85.0);
                        }
                        if q.contains("warm") || q.contains("tan") || q.contains("golden") || q.contains("sun") {
                            r.skin_warmth = (r.skin_warmth + 18.0).clamp(15.0, 50.0);
                            let cur_rel = k_to_rel(dd.wb.temp);
                            dd.wb.temp = rel_to_k(cur_rel + 16.0);
                            dd.wb.mode = lightcraft_develop::WbMode::Custom;
                        }
                        if q.contains("cool") || q.contains("cold") || q.contains("icy") {
                            let cur_rel = k_to_rel(dd.wb.temp);
                            dd.wb.temp = rel_to_k(cur_rel - 16.0);
                            dd.wb.mode = lightcraft_develop::WbMode::Custom;
                        }
                        if q.contains("red") || q.contains("rosacea") || q.contains("flush") {
                            r.skin_redness = (r.skin_redness - 20.0).clamp(-55.0, -15.0);
                        }
                        if q.contains("eye") || q.contains("catchlight") {
                            r.eye_brighten = (r.eye_brighten + 20.0).clamp(42.0, 80.0);
                            r.eye_whiten = (r.eye_whiten + 15.0).clamp(30.0, 70.0);
                            r.eye_clarity = (r.eye_clarity + 18.0).clamp(35.0, 75.0);
                        }
                        if q.contains("teeth") || q.contains("whiten") || q.contains("smile") {
                            r.teeth_whiten = (r.teeth_whiten + 22.0).clamp(45.0, 85.0);
                            r.smile_lift = (r.smile_lift + 15.0).clamp(18.0, 55.0);
                        }
                        if q.contains("lip") || q.contains("gloss") {
                            r.lip_enhance = (r.lip_enhance + 18.0).clamp(25.0, 65.0);
                            r.lip_moisture = (r.lip_moisture + 22.0).clamp(30.0, 75.0);
                        }
                        if q.contains("jaw") || q.contains("sculpt") || q.contains("contour") || q.contains("slim face") || q.contains("face slim") {
                            r.face_contour = (r.face_contour + 20.0).clamp(40.0, 80.0);
                            r.face_slim = (r.face_slim + 18.0).clamp(20.0, 60.0);
                        }
                        if q.contains("body") || q.contains("slim") || q.contains("thin") || q.contains("skinny") || q.contains("weight") || q.contains("fit") {
                            r.body_slim = (r.body_slim + 28.0).clamp(25.0, 75.0);
                            r.waist_slim = (r.waist_slim + 26.0).clamp(25.0, 75.0);
                            r.arm_slim = (r.arm_slim + 22.0).clamp(20.0, 65.0);
                        }
                        if q.contains("stomach") || q.contains("belly") || q.contains("tummy") || q.contains("abdomen") || q.contains("gut") || q.contains("bloat") || q.contains("flat") {
                            r.stomach_flat = (r.stomach_flat + 42.0).clamp(40.0, 90.0);
                            r.waist_slim = (r.waist_slim + 28.0).clamp(30.0, 80.0);
                        }
                        if q.contains("waist") || q.contains("hourglass") || q.contains("cinch") {
                            r.waist_slim = (r.waist_slim + 34.0).clamp(35.0, 85.0);
                            r.hip_enhance = (r.hip_enhance + 18.0).clamp(15.0, 60.0);
                        }
                        if q.contains("hip") || q.contains("curve") || q.contains("s-curve") || q.contains("booty") || q.contains("glute") {
                            if q.contains("slim hip") || q.contains("narrow hip") || q.contains("reduce hip") {
                                r.hip_enhance = (r.hip_enhance - 28.0).clamp(-75.0, -20.0);
                            } else {
                                r.hip_enhance = (r.hip_enhance + 28.0).clamp(25.0, 75.0);
                            }
                        }
                        if q.contains("shoulder") || q.contains("posture") {
                            if q.contains("narrow") || q.contains("slim shoulder") || q.contains("smaller shoulder") {
                                r.shoulder_width = (r.shoulder_width - 25.0).clamp(-70.0, -18.0);
                            } else {
                                r.shoulder_width = (r.shoulder_width + 25.0).clamp(20.0, 70.0);
                            }
                        }
                        if q.contains("arm") || q.contains("bicep") || q.contains("tricep") {
                            r.arm_slim = (r.arm_slim + 32.0).clamp(28.0, 80.0);
                        }
                        if q.contains("neck") || q.contains("collarbone") || q.contains("clavicle") || q.contains("double chin") {
                            r.neck_slim = (r.neck_slim + 30.0).clamp(25.0, 75.0);
                        }
                        if q.contains("chest") || q.contains("bust") || q.contains("cleavage") || q.contains("pec") {
                            if q.contains("reduce") || q.contains("smaller") || q.contains("flat chest") {
                                r.chest_enhance = (r.chest_enhance - 28.0).clamp(-75.0, -20.0);
                            } else {
                                r.chest_enhance = (r.chest_enhance + 28.0).clamp(25.0, 75.0);
                            }
                        }
                        if q.contains("leg") || q.contains("tall") || q.contains("height") || q.contains("thigh") || q.contains("calf") || q.contains("lengthen") {
                            r.leg_length = (r.leg_length + 30.0).clamp(25.0, 80.0);
                        }
                        if q.contains("nose") {
                            r.nose_slim = (r.nose_slim + 18.0).clamp(20.0, 55.0);
                        }
                        if q.contains("chin") && !q.contains("double chin") {
                            r.chin_length = -18.0;
                        }
                        if q.contains("hair") || q.contains("flyaway") || q.contains("stray") {
                            r.hair_luster = (r.hair_luster + 22.0).clamp(35.0, 75.0);
                            r.stray_hair = (r.stray_hair + 26.0).clamp(40.0, 85.0);
                        }
                        if q.contains("auto tone") || q.contains("balance exposure") {
                            dd.light.exposure = (dd.light.exposure + 0.25).clamp(-4.0, 4.0);
                            dd.light.highlights = -24.0;
                            dd.light.shadows = 28.0;
                            dd.light.contrast = 12.0;
                            dd.color.vibrance = 14.0;
                        }
                        if q.contains("brighten") || q.contains("brighter") || q.contains("increase exposure") {
                            dd.light.exposure = (dd.light.exposure + 0.45).clamp(-4.0, 4.0);
                            dd.light.shadows = (dd.light.shadows + 18.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("darken") && !q.contains("background") {
                            dd.light.exposure = (dd.light.exposure - 0.40).clamp(-4.0, 4.0);
                        }
                        if q.contains("darken") && q.contains("background") {
                            dd.vignette.amount = (dd.vignette.amount - 34.0).clamp(-100.0, 100.0);
                            dd.light.highlights = (dd.light.highlights - 15.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("lift shadow") || q.contains("open shadow") || q.contains("shadows") {
                            dd.light.shadows = (dd.light.shadows + 28.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("highlight") {
                            dd.light.highlights = (dd.light.highlights - 28.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("contrast") || q.contains("punch") || q.contains("pop") {
                            dd.light.contrast = (dd.light.contrast + 20.0).clamp(-100.0, 100.0);
                            dd.effects.clarity = (dd.effects.clarity + 12.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("cinematic") || q.contains("film") || q.contains("teal") {
                            dd.light.contrast = 18.0;
                            dd.light.highlights = -22.0;
                            dd.light.shadows = 20.0;
                            dd.color.vibrance = 16.0;
                            dd.color.saturation = -6.0;
                            dd.grading.shadows.hue = 204.0;
                            dd.grading.shadows.sat = 20.0;
                            dd.grading.highlights.hue = 38.0;
                            dd.grading.highlights.sat = 18.0;
                            dd.vignette.amount = -22.0;
                            dd.grain.amount = 14.0;
                        }
                        if q.contains("black and white") || q.contains("b&w") || q.contains("monochrome") || q.contains("bw") {
                            dd.treatment = lightcraft_develop::Treatment::Bw;
                            dd.light.contrast = (dd.light.contrast + 18.0).clamp(-100.0, 100.0);
                            dd.effects.clarity = (dd.effects.clarity + 15.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("color mode") || q.contains("restore color") {
                            dd.treatment = lightcraft_develop::Treatment::Color;
                        }
                        if q.contains("vibrant") || q.contains("colorful") || q.contains("saturat") {
                            dd.color.vibrance = (dd.color.vibrance + 24.0).clamp(-100.0, 100.0);
                            dd.color.saturation = (dd.color.saturation + 10.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("desaturat") || q.contains("muted") || q.contains("pastel") {
                            dd.color.saturation = (dd.color.saturation - 22.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("sharp") || q.contains("crisp") || q.contains("detail") {
                            dd.detail.sharpen_amount = (dd.detail.sharpen_amount + 22.0).clamp(0.0, 150.0);
                            dd.effects.texture = (dd.effects.texture + 14.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("denoise") || q.contains("noise") {
                            dd.detail.nr_luminance = (dd.detail.nr_luminance + 32.0).clamp(0.0, 100.0);
                        }
                        if q.contains("dehaze") || q.contains("haze") || q.contains("fog") {
                            dd.effects.dehaze = (dd.effects.dehaze + 25.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("vignette") {
                            dd.vignette.amount = (dd.vignette.amount - 26.0).clamp(-100.0, 100.0);
                        }
                        if q.contains("grain") {
                            dd.grain.amount = (dd.grain.amount + 22.0).clamp(0.0, 100.0);
                        }
                        let crop_aspect: Option<(u32, u32)> = if q.contains("1:1") || q.contains("1x1") || q.contains("square") {
                            Some((100, 100))
                        } else if q.contains("4:5") || q.contains("4x5") || q.contains("5:4") || q.contains("5x4") {
                            Some((400, 500))
                        } else if q.contains("16:9") || q.contains("16x9") {
                            Some((1600, 900))
                        } else if q.contains("9:16") || q.contains("9x16") {
                            Some((900, 1600))
                        } else if q.contains("3:2") || q.contains("3x2") {
                            Some((300, 200))
                        } else if q.contains("5:7") || q.contains("5x7") {
                            Some((500, 700))
                        } else {
                            None
                        };
                        if let Some((aw, ah)) = crop_aspect {
                            let ph = s.catalog.photo(id);
                            let (mut pw, mut ph_h) = ph.map(|p| (p.width.max(1) as f64, p.height.max(1) as f64)).unwrap_or((3.0, 2.0));
                            if dd.orientation.swaps_axes() {
                                std::mem::swap(&mut pw, &mut ph_h);
                            }
                            dd.crop.aspect = Some((aw, ah));
                            let ratio = if (pw >= ph_h) == (aw >= ah) {
                                aw as f64 / ah as f64
                            } else {
                                ah as f64 / aw as f64
                            };
                            dd.crop.geometry = lightcraft_geom::crop_fit_angle(pw, ph_h, dd.crop.geometry.angle, Some(ratio));
                        }
                        if q.contains("flip horizontal") || q.contains("mirror") {
                            dd.crop.flip_h = !dd.crop.flip_h;
                        }
                        if q.contains("flip vertical") {
                            dd.crop.flip_v = !dd.crop.flip_v;
                        }
                    }

                    // If AI Vision returned structured JSON overrides, apply ONLY the fields present:
                    if let Some(ai) = &parsed_ai {
                        if actions.iter().any(|a| a == "reset") {
                            dd = lightcraft_develop::DevelopSettings::default();
                            r = lightcraft_develop::Retouch::default();
                        }
                        let num = |obj: &Value, k1: &str, k2: &str| -> Option<f64> {
                            obj.get(k1).or_else(|| obj.get(k2)).and_then(Value::as_f64).filter(|v| v.is_finite())
                        };
                        if let Some(ret_obj) = ai.get("retouch").or(Some(ai)) {
                            if let Some(v) = num(ret_obj, "skinSmooth", "skin_smooth") { r.skin_smooth = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "skinTexture", "skin_texture") { r.skin_texture = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "skinToneEven", "skin_tone_even") { r.skin_tone_even = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "shineControl", "shine_control") { r.shine_control = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "darkCircles", "dark_circles") { r.dark_circles = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "wrinkleReduce", "wrinkle_reduce") { r.wrinkle_reduce = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "skinRadiance", "skin_radiance") { r.skin_radiance = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "skinWarmth", "skin_warmth") { r.skin_warmth = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "skinRedness", "skin_redness") { r.skin_redness = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "eyeBrighten", "eye_brighten") { r.eye_brighten = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "eyeWhiten", "eye_whiten") { r.eye_whiten = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "eyeClarity", "eye_clarity") { r.eye_clarity = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "eyebrowEnhance", "eyebrow_enhance") { r.eyebrow_enhance = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "teethWhiten", "teeth_whiten") { r.teeth_whiten = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "lipEnhance", "lip_enhance") { r.lip_enhance = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "lipMoisture", "lip_moisture") { r.lip_moisture = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "faceContour", "face_contour") { r.face_contour = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "faceSlim", "face_slim") { r.face_slim = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "eyeEnlarge", "eye_enlarge") { r.eye_enlarge = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "noseSlim", "nose_slim") { r.nose_slim = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "chinLength", "chin_length") { r.chin_length = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "smileLift", "smile_lift") { r.smile_lift = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "bodySlim", "body_slim") { r.body_slim = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "waistSlim", "waist_slim") { r.waist_slim = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "stomachFlat", "stomach_flat") { r.stomach_flat = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "hipEnhance", "hip_enhance") { r.hip_enhance = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "shoulderWidth", "shoulder_width") { r.shoulder_width = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "armSlim", "arm_slim") { r.arm_slim = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "neckSlim", "neck_slim") { r.neck_slim = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "chestEnhance", "chest_enhance") { r.chest_enhance = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "legLength", "leg_length") { r.leg_length = v.clamp(-100.0, 100.0); }
                            if let Some(v) = num(ret_obj, "hairLuster", "hair_luster") { r.hair_luster = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "strayHair", "stray_hair") { r.stray_hair = v.clamp(0.0, 100.0); }
                            if let Some(v) = num(ret_obj, "blemishSensitivity", "blemish_sensitivity") { r.blemish_sensitivity = v.clamp(0.0, 100.0); }
                        }

                        let light_obj = ai.get("light").unwrap_or(ai);
                        if let Some(v) = num(light_obj, "exposure", "exposure") { dd.light.exposure = v.clamp(-5.0, 5.0); }
                        if let Some(v) = num(light_obj, "contrast", "contrast") { dd.light.contrast = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(light_obj, "highlights", "highlights") { dd.light.highlights = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(light_obj, "shadows", "shadows") { dd.light.shadows = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(light_obj, "whites", "whites") { dd.light.whites = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(light_obj, "blacks", "blacks") { dd.light.blacks = v.clamp(-100.0, 100.0); }

                        let wb_obj = ai.get("wb").unwrap_or(ai);
                        if let Some(v) = num(wb_obj, "temp", "temperature") {
                            dd.wb.temp = if v >= 1500.0 { v.clamp(2000.0, 50000.0) } else { rel_to_k(v) };
                            dd.wb.mode = lightcraft_develop::WbMode::Custom;
                        }
                        if let Some(v) = num(wb_obj, "tint", "tint") {
                            dd.wb.tint = v.clamp(-150.0, 150.0);
                            dd.wb.mode = lightcraft_develop::WbMode::Custom;
                        }

                        let color_obj = ai.get("color").unwrap_or(ai);
                        if let Some(v) = num(color_obj, "vibrance", "vibrance") { dd.color.vibrance = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(color_obj, "saturation", "saturation") { dd.color.saturation = v.clamp(-100.0, 100.0); }

                        if let Some(gr) = ai.get("grading") {
                            let apply_wheel = |w: &mut lightcraft_develop::Wheel, src: &Value| {
                                if let Some(h) = num(src, "hue", "h") { w.hue = h.rem_euclid(360.0); }
                                if let Some(s) = num(src, "sat", "saturation") { w.sat = s.clamp(0.0, 100.0); }
                                if let Some(l) = num(src, "lum", "luminance") { w.lum = l.clamp(-100.0, 100.0); }
                            };
                            if let Some(sh) = gr.get("shadows") { apply_wheel(&mut dd.grading.shadows, sh); }
                            if let Some(mid) = gr.get("midtones") { apply_wheel(&mut dd.grading.midtones, mid); }
                            if let Some(hi) = gr.get("highlights") { apply_wheel(&mut dd.grading.highlights, hi); }
                            if let Some(gl) = gr.get("global") { apply_wheel(&mut dd.grading.global, gl); }
                            if let Some(h) = num(gr, "shadowsHue", "shadows_hue") { dd.grading.shadows.hue = h.rem_euclid(360.0); }
                            if let Some(s) = num(gr, "shadowsSat", "shadows_sat") { dd.grading.shadows.sat = s.clamp(0.0, 100.0); }
                            if let Some(l) = num(gr, "shadowsLum", "shadows_lum") { dd.grading.shadows.lum = l.clamp(-100.0, 100.0); }
                            if let Some(h) = num(gr, "midtonesHue", "midtones_hue") { dd.grading.midtones.hue = h.rem_euclid(360.0); }
                            if let Some(s) = num(gr, "midtonesSat", "midtones_sat") { dd.grading.midtones.sat = s.clamp(0.0, 100.0); }
                            if let Some(l) = num(gr, "midtonesLum", "midtones_lum") { dd.grading.midtones.lum = l.clamp(-100.0, 100.0); }
                            if let Some(h) = num(gr, "highlightsHue", "highlights_hue") { dd.grading.highlights.hue = h.rem_euclid(360.0); }
                            if let Some(s) = num(gr, "highlightsSat", "highlights_sat") { dd.grading.highlights.sat = s.clamp(0.0, 100.0); }
                            if let Some(l) = num(gr, "highlightsLum", "highlights_lum") { dd.grading.highlights.lum = l.clamp(-100.0, 100.0); }
                        }

                        if let Some(mx) = ai.get("mixer") {
                            let apply_band = |b: &mut lightcraft_develop::Hsl, src: &Value| {
                                if let Some(h) = num(src, "hue", "h") { b.hue = h.clamp(-100.0, 100.0); }
                                if let Some(s) = num(src, "sat", "saturation") { b.sat = s.clamp(-100.0, 100.0); }
                                if let Some(l) = num(src, "lum", "luminance") { b.lum = l.clamp(-100.0, 100.0); }
                            };
                            if let Some(v) = mx.get("red") { apply_band(&mut dd.mixer.red, v); }
                            if let Some(v) = mx.get("orange") { apply_band(&mut dd.mixer.orange, v); }
                            if let Some(v) = mx.get("yellow") { apply_band(&mut dd.mixer.yellow, v); }
                            if let Some(v) = mx.get("green") { apply_band(&mut dd.mixer.green, v); }
                            if let Some(v) = mx.get("aqua") { apply_band(&mut dd.mixer.aqua, v); }
                            if let Some(v) = mx.get("blue") { apply_band(&mut dd.mixer.blue, v); }
                            if let Some(v) = mx.get("purple") { apply_band(&mut dd.mixer.purple, v); }
                            if let Some(v) = mx.get("magenta") { apply_band(&mut dd.mixer.magenta, v); }
                        }

                        let effects_obj = ai.get("effects").unwrap_or(ai);
                        if let Some(v) = num(effects_obj, "texture", "texture") { dd.effects.texture = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(effects_obj, "clarity", "clarity") { dd.effects.clarity = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(effects_obj, "dehaze", "dehaze") { dd.effects.dehaze = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(effects_obj, "vignette", "vignette") { dd.vignette.amount = v.clamp(-100.0, 100.0); }
                        if let Some(v) = num(effects_obj, "grain", "grain") { dd.grain.amount = v.clamp(0.0, 100.0); }

                        let detail_obj = ai.get("detail").unwrap_or(ai);
                        if let Some(v) = num(detail_obj, "sharpness", "sharpenAmount") { dd.detail.sharpen_amount = v.clamp(0.0, 150.0); }
                        if let Some(v) = num(detail_obj, "nrLuminance", "noise") { dd.detail.nr_luminance = v.clamp(0.0, 100.0); }
                        if let Some(v) = num(detail_obj, "nrColor", "colorNoise") { dd.detail.nr_color = v.clamp(0.0, 100.0); }

                        if let Some(cr) = ai.get("crop") {
                            if let Some(asp) = cr.get("aspect").and_then(Value::as_str) {
                                let pair = match asp.trim().to_ascii_lowercase().as_str() {
                                    "1:1" | "1x1" | "square" => Some((100u32, 100u32)),
                                    "4:5" | "4x5" | "5:4" | "5x4" => Some((400, 500)),
                                    "16:9" | "16x9" => Some((1600, 900)),
                                    "9:16" | "9x16" => Some((900, 1600)),
                                    "3:2" | "3x2" => Some((300, 200)),
                                    "5:7" | "5x7" => Some((500, 700)),
                                    "original" | "free" | "none" => {
                                        dd.crop.aspect = None;
                                        None
                                    }
                                    _ => None,
                                };
                                if let Some((aw, ah)) = pair {
                                    let ph = s.catalog.photo(id);
                                    let (mut pw, mut ph_h) = ph.map(|p| (p.width.max(1) as f64, p.height.max(1) as f64)).unwrap_or((3.0, 2.0));
                                    if dd.orientation.swaps_axes() {
                                        std::mem::swap(&mut pw, &mut ph_h);
                                    }
                                    dd.crop.aspect = Some((aw, ah));
                                    let ratio = if (pw >= ph_h) == (aw >= ah) {
                                        aw as f64 / ah as f64
                                    } else {
                                        ah as f64 / aw as f64
                                    };
                                    dd.crop.geometry = lightcraft_geom::crop_fit_angle(pw, ph_h, dd.crop.geometry.angle, Some(ratio));
                                }
                            }
                            if let Some(ang) = num(cr, "angle", "straighten") {
                                let ph = s.catalog.photo(id);
                                let (mut pw, mut ph_h) = ph.map(|p| (p.width.max(1) as f64, p.height.max(1) as f64)).unwrap_or((3.0, 2.0));
                                if dd.orientation.swaps_axes() {
                                    std::mem::swap(&mut pw, &mut ph_h);
                                }
                                let ratio = dd.crop.aspect.map(|(aw, ah)| aw as f64 / ah as f64);
                                dd.crop.geometry = lightcraft_geom::crop_fit_angle(pw, ph_h, ang.clamp(-45.0, 45.0), ratio);
                            }
                            if let Some(fh) = cr.get("flipH").and_then(Value::as_bool) { dd.crop.flip_h = fh; }
                            if let Some(fv) = cr.get("flipV").and_then(Value::as_bool) { dd.crop.flip_v = fv; }
                        }

                        if let Some(tr) = ai.get("treatment").and_then(Value::as_str) {
                            if tr.eq_ignore_ascii_case("bw") || tr.eq_ignore_ascii_case("monochrome") {
                                dd.treatment = lightcraft_develop::Treatment::Bw;
                            } else if tr.eq_ignore_ascii_case("color") {
                                dd.treatment = lightcraft_develop::Treatment::Color;
                            }
                        }

                        if actions.iter().any(|a| a == "clearmasks" || a == "clear_masks") {
                            dd.masks.clear();
                        }
                        if actions.iter().any(|a| a == "clearspots" || a == "clear_spots") {
                            dd.spots.clear();
                        }

                        if let Some(mask_arr) = ai.get("masks").and_then(Value::as_array) {
                            for m_val in mask_arr.iter().take(6) {
                                let kind = m_val.get("kind").and_then(Value::as_str).unwrap_or("background").trim().to_ascii_lowercase();
                                let Ok(shape) = shape_from(&kind, m_val, "retouch.aiApply") else { continue };
                                let default_name = match kind.as_str() {
                                    "background" => "AI Background",
                                    "sky" => "AI Sky",
                                    "subject" => "AI Subject",
                                    "people" => "AI Portrait Mask",
                                    "radial" => "AI Radial Mask",
                                    "linear" => "AI Gradient Mask",
                                    _ => "AI Mask",
                                };
                                let name = m_val
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .map(str::trim)
                                    .filter(|n| !n.is_empty())
                                    .unwrap_or(default_name)
                                    .to_string();
                                let invert = m_val.get("invert").and_then(Value::as_bool).unwrap_or(false);
                                let adj_src = m_val.get("adjust").unwrap_or(m_val);
                                let mut adj = LocalAdjustments::default();
                                if let Some(v) = num(adj_src, "exposure", "exposure") { adj.exposure = v; }
                                if let Some(v) = num(adj_src, "contrast", "contrast") { adj.contrast = v; }
                                if let Some(v) = num(adj_src, "highlights", "highlights") { adj.highlights = v; }
                                if let Some(v) = num(adj_src, "shadows", "shadows") { adj.shadows = v; }
                                if let Some(v) = num(adj_src, "whites", "whites") { adj.whites = v; }
                                if let Some(v) = num(adj_src, "blacks", "blacks") { adj.blacks = v; }
                                if let Some(v) = num(adj_src, "temp", "temperature") { adj.temp = v; }
                                if let Some(v) = num(adj_src, "tint", "tint") { adj.tint = v; }
                                if let Some(v) = num(adj_src, "texture", "texture") { adj.texture = v; }
                                if let Some(v) = num(adj_src, "clarity", "clarity") { adj.clarity = v; }
                                if let Some(v) = num(adj_src, "dehaze", "dehaze") { adj.dehaze = v; }
                                if let Some(v) = num(adj_src, "saturation", "sat") { adj.saturation = v; }
                                if let Some(v) = num(adj_src, "sharpness", "sharp") { adj.sharpness = v; }
                                if let Some(v) = num(adj_src, "noise", "denoise") { adj.noise = v; }
                                let adj = clamp_local(adj);
                                if let Some(existing) = dd.masks.iter_mut().find(|m| m.name.eq_ignore_ascii_case(&name)) {
                                    existing.adjust = adj;
                                    existing.invert = invert;
                                    existing.visible = true;
                                } else {
                                    let mid = dd.next_mask_id();
                                    dd.masks.push(Mask {
                                        id: mid,
                                        name,
                                        visible: true,
                                        invert,
                                        refine: 25.0,
                                        components: vec![MaskComponent {
                                            name: None,
                                            op: MaskOp::Add,
                                            invert: false,
                                            shape,
                                        }],
                                        adjust: adj,
                                    });
                                    s.active_mask = Some(mid);
                                }
                            }
                        }
                    }

                    dd.retouch = r;

                    if id == active_id {
                        diff_notes = describe_develop_diff(&prev_dd, &dd);
                    }

                    if dd != prev_dd
                        && let Some(op) = s.develop_op(id, dd, "AI Studio Assistant")
                    {
                        ops.push(op);
                    }
                }

                let changed = ops.len();
                if !ops.is_empty() {
                    s.commit("AI Studio Assistant", lightcraft_catalog::Op::Batch { ops })?;
                }

                if summary.is_empty() {
                    if !diff_notes.is_empty() {
                        summary = format!("Adjusted {}.", diff_notes.join(" · "));
                    } else {
                        summary = "Ready — tell me what you'd like to adjust or ask anything about the photo.".to_string();
                    }
                } else if !diff_notes.is_empty() {
                    summary = format!("{summary}\n\n✓ {}", diff_notes.join(" · "));
                }

                Ok(json!({
                    "changed": changed,
                    "summary": summary,
                    "diff": diff_notes,
                    "neural": any_neural,
                }))
            }
        ),
        cmd!(
            "spot.add",
            "Add Remove Spot",
            [],
            None,
            "{mode?: remove|heal|clone|fill (content-aware), points: [[x,y],…], size?: fraction of long edge, feather?, opacity?, source?: [dx,dy] (default: the best match nearby)} — selects the new spot; returns {index}",
            has_active,
            |s, p| {
                let mode: SpotMode =
                    serde_json::from_value(p.get("mode").cloned().unwrap_or(json!("remove"))).map_err(|e| bad("spot.add", e.to_string()))?;
                let pts: Vec<Point> = p
                    .get("points")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(|q| Some(Point::new(q.get(0)?.as_f64()?, q.get(1)?.as_f64()?))).collect())
                    .unwrap_or_default();
                if pts.is_empty() {
                    return Err(bad("spot.add", "missing points"));
                }
                let d = Spot::default();
                let mut spot = Spot {
                    mode,
                    points: pts,
                    size: f64_or(p, "size", d.size).clamp(SPOT_SIZE.0, SPOT_SIZE.1),
                    feather: f64_or(p, "feather", d.feather).clamp(0.0, 100.0),
                    opacity: f64_or(p, "opacity", d.opacity).clamp(0.0, 100.0),
                    source_offset: point(p, "source"),
                };
                let id = s.active().ok_or_else(|| bad("spot.add", "no active photo"))?;
                let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
                if spot.source_offset.is_none() && spot.mode != SpotMode::Fill {
                    // resolve the automatic source now: it gets a pin and stays put at every size
                    spot.source_offset = pick_source(s, id, &dd, &spot, None);
                }
                dd.spots.push(spot);
                let n = dd.spots.len();
                s.set_develop(id, dd, "Remove")?;
                s.active_spot = Some(n - 1);
                Ok(json!({"index": n - 1}))
            }
        ),
        cmd!("spot.select", "Select Spot", [], None, "{index|null}", has_active, |s, p| {
            let n = spots_len(s);
            s.active_spot = match p.get("index").and_then(Value::as_u64) {
                Some(i) if (i as usize) < n => Some(i as usize),
                Some(_) => return Err(bad("spot.select", "no such spot")),
                None => None,
            };
            Ok(json!({"activeSpot": s.active_spot}))
        }),
        cmd!(
            "spot.update",
            "Edit Spot",
            [],
            None,
            "{index? (the selected spot), move?: [dx,dy] (the target, normalized), source?: [dx,dy] (offset from the target), moveSource?: [dx,dy], size?, feather?, opacity?, mode?}",
            has_active,
            |s, p| {
                let c = "spot.update";
                let i = spot_index(s, p, c)?;
                let mode: Option<SpotMode> = match p.get("mode") {
                    Some(m) => Some(serde_json::from_value(m.clone()).map_err(|e| bad(c, e.to_string()))?),
                    None => None,
                };
                spots_edit(s, c, "Edit Spot", |spots| {
                    let sp = &mut spots[i];
                    if let Some(d) = point(p, "move") {
                        sp.points.iter_mut().for_each(|q| *q = Point::new(q.x + d.x, q.y + d.y));
                        // the source moves along (its offset is relative to the target)
                    }
                    if let Some(o) = point(p, "source") {
                        sp.source_offset = Some(o);
                    }
                    if let Some(d) = point(p, "moveSource") {
                        let o = sp.source_offset.unwrap_or(Point::new(0.0, 0.0));
                        sp.source_offset = Some(Point::new(o.x + d.x, o.y + d.y));
                    }
                    if let Some(v) = p.get("size").and_then(Value::as_f64) {
                        sp.size = v.clamp(SPOT_SIZE.0, SPOT_SIZE.1);
                    }
                    if let Some(v) = p.get("feather").and_then(Value::as_f64) {
                        sp.feather = v.clamp(0.0, 100.0);
                    }
                    if let Some(v) = p.get("opacity").and_then(Value::as_f64) {
                        sp.opacity = v.clamp(0.0, 100.0);
                    }
                    if let Some(m) = mode {
                        sp.mode = m;
                    }
                    Ok(())
                })?;
                Ok(json!({"index": i}))
            }
        ),
        cmd!(
            "spot.refreshSource",
            "Refresh Source",
            [],
            None,
            "{index? (the selected spot)} — picks the next-best source away from the current one",
            has_active,
            |s, p| {
                let c = "spot.refreshSource";
                let i = spot_index(s, p, c)?;
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                let dd = (*s.develop_of(id).unwrap_or_default()).clone();
                let spot = dd.spots[i].clone();
                let next = pick_source(s, id, &dd, &spot, spot.source_offset).ok_or_else(|| bad(c, "no other source fits in the photo"))?;
                spots_edit(s, c, "Refresh Source", |spots| {
                    spots[i].source_offset = Some(next);
                    Ok(())
                })?;
                Ok(json!({"index": i, "source": [next.x, next.y]}))
            }
        ),
        cmd!("spot.delete", "Delete Spot", [], None, "{index? (the selected spot)}", has_active, |s, p| {
            let c = "spot.delete";
            let i = spot_index(s, p, c)?;
            spots_edit(s, c, "Delete Spot", |spots| {
                spots.remove(i);
                Ok(())
            })?;
            s.active_spot = match s.active_spot {
                Some(a) if a == i => None,
                Some(a) if a > i => Some(a - 1),
                a => a,
            };
            ok()
        }),
        // ---- Red eye / pet eye
        cmd!(
            "redeye.add",
            "Add Red Eye Correction",
            [],
            None,
            "{center: [x,y] normalized, rx, ry: radii as fractions of the long edge, pet?: bool, pupilSize?: 0..100, darken?: 0..100} — the pupil inside is found automatically; returns {index}",
            has_active,
            |s, p| {
                let c = "redeye.add";
                let center = point(p, "center").ok_or_else(|| bad(c, "missing center"))?;
                let d = RedEye::default();
                let eye = RedEye {
                    center,
                    rx: f64_or(p, "rx", d.rx).clamp(1e-4, 0.5),
                    ry: f64_or(p, "ry", d.ry).clamp(1e-4, 0.5),
                    pupil_size: f64_or(p, "pupilSize", d.pupil_size).clamp(0.0, 100.0),
                    darken: f64_or(p, "darken", d.darken).clamp(0.0, 100.0),
                    pet: bool_or(p, "pet", false),
                    catchlight: None,
                };
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
                dd.red_eye.push(eye);
                let n = dd.red_eye.len();
                s.set_develop(id, dd, if eye.pet { "Pet Eye" } else { "Red Eye" })?;
                Ok(json!({"index": n - 1}))
            }
        ),
        cmd!(
            "redeye.catchlight",
            "Pet Eye Catchlight",
            [],
            None,
            "{index, on?: bool (default true), offset?: [dx, dy] from the pupil centre in pupil radii (default [-0.35, -0.35])}",
            has_active,
            |s, p| {
                let c = "redeye.catchlight";
                let i = super::index_req(p, "index", c)?;
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
                let eye = dd.red_eye.get_mut(i).ok_or_else(|| bad(c, "no such eye"))?;
                if !eye.pet {
                    return Err(bad(c, "catchlights are for pet eyes"));
                }
                let off = point(p, "offset").or(eye.catchlight).unwrap_or(Point::new(-0.35, -0.35));
                eye.catchlight = bool_or(p, "on", true).then(|| Point::new(off.x.clamp(-1.0, 1.0), off.y.clamp(-1.0, 1.0)));
                s.set_develop(id, dd, "Catchlight")?;
                ok()
            }
        ),
        cmd!("redeye.delete", "Delete Red Eye Correction", [], None, "{index}", has_active, |s, p| {
            let c = "redeye.delete";
            let i = super::index_req(p, "index", c)?;
            let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
            let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
            if i >= dd.red_eye.len() {
                return Err(bad(c, "no such eye"));
            }
            dd.red_eye.remove(i);
            s.set_develop(id, dd, "Delete Red Eye")?;
            ok()
        }),
    ]
}

/// Spot radius limits (fraction of the long edge).
pub const SPOT_SIZE: (f64, f64) = (0.001, 0.25);

fn spots_len(s: &Session) -> usize {
    s.active().and_then(|id| s.develop_of(id)).map(|d| d.spots.len()).unwrap_or(0)
}

/// `index` from the params, else the selected spot.
fn spot_index(s: &Session, p: &Value, c: &str) -> Result<usize> {
    let i = p.get("index").and_then(Value::as_u64).map(|v| v as usize).or(s.active_spot).ok_or_else(|| bad(c, "no spot selected (give `index`)"))?;
    if i >= spots_len(s) {
        return Err(bad(c, "no such spot"));
    }
    Ok(i)
}

fn spots_edit(s: &mut Session, c: &str, label: &str, f: impl FnOnce(&mut Vec<Spot>) -> Result<()>) -> Result<()> {
    let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
    let mut d = (*s.develop_of(id).unwrap_or_default()).clone();
    f(&mut d.spots)?;
    s.set_develop(id, d, label)
}

/// An automatic source for `spot` on photo `id` (a small proxy of the photo, framed by `d`).
fn pick_source(s: &mut Session, id: crate::PhotoId, d: &lightcraft_develop::DevelopSettings, spot: &Spot, avoid: Option<Point>) -> Option<Point> {
    let src = s.source_now(id, crate::media::SourceLevel::Thumb).ok()?;
    let info = s.source_info(id);
    lightcraft_pipeline::spots::pick_source(&src, &info, d, spot, avoid)
}

pub(crate) fn clamp_local(mut a: LocalAdjustments) -> LocalAdjustments {
    let c = |v: &mut f64, lo: f64, hi: f64| *v = if v.is_finite() { v.clamp(lo, hi) } else { 0.0 };
    c(&mut a.exposure, -4.0, 4.0);
    for v in [
        &mut a.temp,
        &mut a.tint,
        &mut a.contrast,
        &mut a.highlights,
        &mut a.shadows,
        &mut a.whites,
        &mut a.blacks,
        &mut a.texture,
        &mut a.clarity,
        &mut a.dehaze,
        &mut a.hue,
        &mut a.saturation,
        &mut a.sharpness,
        &mut a.noise,
        &mut a.moire,
        &mut a.defringe,
    ] {
        c(v, -100.0, 100.0);
    }
    c(&mut a.color_hue, 0.0, 360.0);
    c(&mut a.color_sat, 0.0, 100.0);
    c(&mut a.amount, 0.0, 200.0);
    a
}

/// `mask.brushStroke`'s `pressure`: one number 0..1 per point (clamped), at most one per point; absent or `null`
/// is no pressure data.
fn pressure_list(v: Option<&Value>, points: usize) -> Result<Vec<f32>> {
    let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(Vec::new()) };
    let list = v.as_array().ok_or_else(|| bad("mask.brushStroke", "pressure must be a list of numbers 0..1"))?;
    if list.len() > points {
        return Err(bad("mask.brushStroke", "pressure has more values than points"));
    }
    list.iter()
        .map(|q| q.as_f64().filter(|f| f.is_finite()).map(|f| f.clamp(0.0, 1.0) as f32))
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| bad("mask.brushStroke", "pressure must be a list of numbers 0..1"))
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(n & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn describe_develop_diff(prev: &lightcraft_develop::DevelopSettings, next: &lightcraft_develop::DevelopSettings) -> Vec<String> {
    let mut out = Vec::new();
    fn chk(out: &mut Vec<String>, label: &str, a: f64, b: f64, signed: bool) {
        if (b - a).abs() >= 0.5 {
            if signed {
                out.push(format!("{label} {:+.0}", b));
            } else {
                out.push(format!("{label} {:.0}", b));
            }
        }
    }
    if (next.light.exposure - prev.light.exposure).abs() >= 0.02 {
        out.push(format!("Exposure {:+.2}", next.light.exposure));
    }
    chk(&mut out, "Contrast", prev.light.contrast, next.light.contrast, true);
    chk(&mut out, "Highlights", prev.light.highlights, next.light.highlights, true);
    chk(&mut out, "Shadows", prev.light.shadows, next.light.shadows, true);
    chk(&mut out, "Whites", prev.light.whites, next.light.whites, true);
    chk(&mut out, "Blacks", prev.light.blacks, next.light.blacks, true);
    if (next.wb.temp - prev.wb.temp).abs() >= 25.0 {
        out.push(format!("Temp {:.0}K", next.wb.temp));
    }
    chk(&mut out, "Tint", prev.wb.tint, next.wb.tint, true);
    chk(&mut out, "Vibrance", prev.color.vibrance, next.color.vibrance, true);
    chk(&mut out, "Saturation", prev.color.saturation, next.color.saturation, true);
    chk(&mut out, "Texture", prev.effects.texture, next.effects.texture, true);
    chk(&mut out, "Clarity", prev.effects.clarity, next.effects.clarity, true);
    chk(&mut out, "Dehaze", prev.effects.dehaze, next.effects.dehaze, true);
    chk(&mut out, "Vignette", prev.vignette.amount, next.vignette.amount, true);
    chk(&mut out, "Grain", prev.grain.amount, next.grain.amount, false);
    chk(&mut out, "Sharpness", prev.detail.sharpen_amount, next.detail.sharpen_amount, false);
    chk(&mut out, "Denoise", prev.detail.nr_luminance, next.detail.nr_luminance, false);
    if next.treatment != prev.treatment {
        out.push(match next.treatment {
            lightcraft_develop::Treatment::Bw => "B&W".to_string(),
            lightcraft_develop::Treatment::Color => "Color".to_string(),
        });
    }
    let pr = &prev.retouch;
    let nr = &next.retouch;
    chk(&mut out, "Skin Smooth", pr.skin_smooth, nr.skin_smooth, false);
    chk(&mut out, "Skin Texture", pr.skin_texture, nr.skin_texture, false);
    chk(&mut out, "Even Tone", pr.skin_tone_even, nr.skin_tone_even, false);
    chk(&mut out, "Shine Control", pr.shine_control, nr.shine_control, false);
    chk(&mut out, "Dark Circles", pr.dark_circles, nr.dark_circles, false);
    chk(&mut out, "Wrinkles", pr.wrinkle_reduce, nr.wrinkle_reduce, false);
    chk(&mut out, "Skin Radiance", pr.skin_radiance, nr.skin_radiance, true);
    chk(&mut out, "Skin Warmth", pr.skin_warmth, nr.skin_warmth, true);
    chk(&mut out, "Skin Redness", pr.skin_redness, nr.skin_redness, true);
    chk(&mut out, "Eye Brighten", pr.eye_brighten, nr.eye_brighten, false);
    chk(&mut out, "Eye Whiten", pr.eye_whiten, nr.eye_whiten, false);
    chk(&mut out, "Eye Clarity", pr.eye_clarity, nr.eye_clarity, false);
    chk(&mut out, "Eyebrows", pr.eyebrow_enhance, nr.eyebrow_enhance, true);
    chk(&mut out, "Teeth Whiten", pr.teeth_whiten, nr.teeth_whiten, false);
    chk(&mut out, "Lips", pr.lip_enhance, nr.lip_enhance, true);
    chk(&mut out, "Lip Moisture", pr.lip_moisture, nr.lip_moisture, false);
    chk(&mut out, "Face Sculpt", pr.face_contour, nr.face_contour, false);
    chk(&mut out, "Face Slim", pr.face_slim, nr.face_slim, true);
    chk(&mut out, "Eye Size", pr.eye_enlarge, nr.eye_enlarge, true);
    chk(&mut out, "Nose Slim", pr.nose_slim, nr.nose_slim, true);
    chk(&mut out, "Chin", pr.chin_length, nr.chin_length, true);
    chk(&mut out, "Smile Lift", pr.smile_lift, nr.smile_lift, true);
    chk(&mut out, "Body Slim", pr.body_slim, nr.body_slim, true);
    chk(&mut out, "Waist Slim", pr.waist_slim, nr.waist_slim, true);
    chk(&mut out, "Stomach Flat", pr.stomach_flat, nr.stomach_flat, false);
    chk(&mut out, "Hips", pr.hip_enhance, nr.hip_enhance, true);
    chk(&mut out, "Shoulders", pr.shoulder_width, nr.shoulder_width, true);
    chk(&mut out, "Arm Slim", pr.arm_slim, nr.arm_slim, true);
    chk(&mut out, "Neck Slim", pr.neck_slim, nr.neck_slim, true);
    chk(&mut out, "Chest", pr.chest_enhance, nr.chest_enhance, true);
    chk(&mut out, "Leg Length", pr.leg_length, nr.leg_length, true);
    chk(&mut out, "Hair Luster", pr.hair_luster, nr.hair_luster, false);
    chk(&mut out, "Stray Hair", pr.stray_hair, nr.stray_hair, false);
    if next.grading != prev.grading {
        out.push("Color Grade".to_string());
    }
    if next.mixer != prev.mixer {
        out.push("Color Mixer".to_string());
    }
    if next.masks != prev.masks {
        if let Some(m) = next.masks.last() {
            out.push(format!("Mask ({})", m.name));
        } else {
            out.push("Cleared Masks".to_string());
        }
    }
    if next.crop.aspect != prev.crop.aspect
        && let Some((w, h)) = next.crop.aspect
    {
        out.push(format!("Crop {w}:{h}"));
    }
    out.truncate(8);
    out
}
