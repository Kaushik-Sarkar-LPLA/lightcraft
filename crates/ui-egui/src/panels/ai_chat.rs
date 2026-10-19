//! Always-on floating AI Studio Chat overlay: conversational control over every setting in the
//! image (Light, Color, White Balance, Grading, Mixer, Effects, Detail, Crop, Masks, Blemish
//! Healing, and all 25 Evoto Portrait Retouch sliders).

use egui::{Align2, Color32, CornerRadius, Margin, RichText, Sense, Stroke, pos2, vec2};
use serde_json::json;

use crate::LightcraftApp;
use crate::state::AiChatMessage;
use crate::theme::Tokens;

/// Trigger an AI image edit / chat turn from either the floating chat or the Retouch panel.
pub fn send_message(app: &mut LightcraftApp, ctx: &egui::Context, prompt: &str, all_selected: bool) {
    let clean_prompt = prompt.trim().to_string();
    let user_display = if clean_prompt.is_empty() {
        if all_selected {
            crate::i18n::tr("✨ Batch Auto-Retouch").to_string()
        } else {
            crate::i18n::tr("✨ Auto-Retouch portrait").to_string()
        }
    } else {
        clean_prompt.clone()
    };

    app.ui.ai_messages.push(AiChatMessage {
        role: "user".into(),
        text: user_display,
    });

    if app.session.active().is_none() {
        let first_id = app.session.catalog.photos().next().map(|p| p.id);
        if let Some(first) = first_id {
            let _ = app.run("photo.select", json!({"id": first.0}));
        } else {
            let msg = crate::i18n::tr("Import a photo first, then chat with me to retouch or edit anything in the image.").to_string();
            app.ui.ai_messages.push(AiChatMessage { role: "assistant".into(), text: msg.clone() });
            app.toast(ctx, msg);
            return;
        }
    }

    // Handle direct undo/redo commands immediately.
    let lower = clean_prompt.to_ascii_lowercase();
    if lower == "undo" || lower == "undo last" {
        match app.run("edit.undo", json!({})) {
            Ok(_) => {
                let msg = crate::i18n::tr("Undid the last edit.").to_string();
                app.ui.ai_summary = msg.clone();
                app.ui.ai_messages.push(AiChatMessage { role: "assistant".into(), text: msg.clone() });
                app.toast(ctx, msg);
            }
            Err(e) => {
                app.ui.ai_messages.push(AiChatMessage { role: "assistant".into(), text: e.to_string() });
                app.toast(ctx, e);
            }
        }
        return;
    }
    if lower == "redo" {
        match app.run("edit.redo", json!({})) {
            Ok(_) => {
                let msg = crate::i18n::tr("Redid the edit.").to_string();
                app.ui.ai_summary = msg.clone();
                app.ui.ai_messages.push(AiChatMessage { role: "assistant".into(), text: msg.clone() });
                app.toast(ctx, msg);
            }
            Err(e) => {
                app.ui.ai_messages.push(AiChatMessage { role: "assistant".into(), text: e.to_string() });
                app.toast(ctx, e);
            }
        }
        return;
    }

    // If the user asked to heal/remove blemishes/acne/spots in chat, run PatchMatch blemish healing too.
    let mut blemish_note = String::new();
    if lower.contains("blemish") || lower.contains("acne") || lower.contains("pimple") || lower.contains("heal spot") {
        if let Ok(r) = app.run("spot.findBlemishes", json!({"allSelected": all_selected})) {
            let n = r["added"].as_u64().unwrap_or(0);
            if n > 0 {
                blemish_note = format!(" Healed {n} blemish{}.", if n == 1 { "" } else { "es" });
            }
        }
    }

    // Apply immediate local intent + YuNet ONNX + Oklab analysis first so the canvas updates with zero latency.
    match app.run("retouch.aiApply", json!({"prompt": clean_prompt, "allSelected": all_selected})) {
        Ok(r) => {
            let mut summary = r["summary"].as_str().unwrap_or("Applied AI adjustments").to_string();
            if !blemish_note.is_empty() {
                summary.push_str(&blemish_note);
            }
            app.ui.ai_summary = summary.clone();
            app.ui.ai_messages.push(AiChatMessage {
                role: "assistant".into(),
                text: summary.clone(),
            });
            app.toast(ctx, summary);
        }
        Err(e) => {
            let err_str = e.to_string();
            app.ui.ai_messages.push(AiChatMessage {
                role: "assistant".into(),
                text: err_str,
            });
            app.toast(ctx, e);
            return;
        }
    }

    // Capture current preview + telemetry and dispatch to the server-side AI Vision endpoint on Web.
    if cfg!(target_arch = "wasm32")
        && let Ok(telemetry) = app.run("retouch.analyze", json!({}))
    {
        let history: Vec<serde_json::Value> = app
            .ui
            .ai_messages
            .iter()
            .rev()
            .take(8)
            .rev()
            .map(|m| json!({"role": m.role, "content": m.text}))
            .collect();
        app.ui.ai_busy = true;
        app.ui.ai_pending_request = Some(json!({
            "endpoint": app.ui.ai_endpoint.trim(),
            "model": app.ui.ai_model.trim(),
            "prompt": clean_prompt,
            "allSelected": all_selected,
            "history": history,
            "telemetry": telemetry,
        }));
    }
}

/// Draw the always-on floating AI Studio Chat over the photo canvas.
pub fn show(app: &mut LightcraftApp, ctx: &egui::Context) {
    // Keep existing automated headless tests clear unless they explicitly opt into testing the chat.
    if app.headless_host && !app.ui.ai_chat_test {
        return;
    }

    let bounds = app.canvas_rect.unwrap_or_else(|| ctx.content_rect());
    if bounds.width() < 180.0 || bounds.height() < 120.0 {
        return;
    }
    let t = Tokens::get(ctx);

    // Minimized state: always-visible floating pill in the bottom-right of the photo canvas.
    if !app.ui.ai_chat_open {
        let pill_pos = pos2(bounds.right() - 16.0, bounds.bottom() - 16.0);
        egui::Area::new(egui::Id::new("ai_studio_chat_pill"))
            .order(egui::Order::Foreground)
            .pivot(Align2::RIGHT_BOTTOM)
            .fixed_pos(pill_pos)
            .show(ctx, |ui| {
                let frame = egui::Frame::NONE
                    .fill(Color32::from_rgba_unmultiplied(24, 27, 34, 240))
                    .corner_radius(CornerRadius::same(18))
                    .stroke(Stroke::new(1.0, t.accent))
                    .inner_margin(Margin { left: 14, right: 14, top: 8, bottom: 8 });
                frame.show(ui, |ui| {
                    let label = if app.ui.ai_busy {
                        crate::i18n::tr("✨ AI Thinking…")
                    } else {
                        crate::i18n::tr("✨ AI Chat")
                    };
                    let resp = ui.add(
                        egui::Button::new(RichText::new(label).size(12.5).strong().color(Color32::from_rgb(235, 240, 250)))
                            .frame(false),
                    );
                    if resp.clicked() {
                        app.ui.ai_chat_open = true;
                        app.ui.ai_chat_focus = true;
                    }
                });
            });
        return;
    }

    // Expanded state: floating draggable AI Studio Chat card constrained to the canvas area.
    let card_w = 336.0f32.min((bounds.width() - 20.0).max(220.0));
    let default_x = (bounds.right() - card_w - 16.0).max(bounds.left() + 8.0);
    let default_y = (bounds.bottom() - 348.0).max(bounds.top() + 8.0);

    let window_frame = egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(22, 24, 30, 244))
        .corner_radius(CornerRadius::same(10))
        .stroke(Stroke::new(1.0, Color32::from_rgb(62, 70, 88)))
        .inner_margin(Margin { left: 12, right: 12, top: 10, bottom: 10 });

    egui::Window::new("ai_studio_chat_window")
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .fixed_size(vec2(card_w, 326.0))
        .default_pos(pos2(default_x, default_y))
        .constrain_to(bounds)
        .order(egui::Order::Foreground)
        .frame(window_frame)
        .show(ctx, |ui| {
            let sel_n = app.session.selection.ids.len();

            // 1. Header bar
            ui.horizontal(|ui| {
                let dot_color = if app.ui.ai_busy {
                    t.accent
                } else {
                    Color32::from_rgb(72, 208, 136)
                };
                let (dot_rect, _) = ui.allocate_exact_size(vec2(8.0, 8.0), Sense::hover());
                ui.painter().circle_filled(dot_rect.center(), 4.0, dot_color);

                ui.label(
                    RichText::new(crate::i18n::tr("✨ AI Studio Assistant"))
                        .size(13.0)
                        .strong()
                        .color(Color32::from_rgb(238, 242, 250)),
                );

                if sel_n > 1 {
                    ui.label(
                        RichText::new(format!("Batch ({sel_n})"))
                            .size(11.0)
                            .color(t.accent),
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(RichText::new("—").size(12.0).color(t.text_dim)).frame(false))
                        .on_hover_text(crate::i18n::tr("Minimize to floating pill"))
                        .clicked()
                    {
                        app.ui.ai_chat_open = false;
                    }
                    if !app.ui.ai_messages.is_empty()
                        && ui
                            .add(egui::Button::new(RichText::new(crate::i18n::tr("Clear")).size(11.0).color(t.text_dim)).frame(false))
                            .on_hover_text(crate::i18n::tr("Clear chat history"))
                            .clicked()
                    {
                        app.ui.ai_messages.clear();
                    }
                });
            });

            ui.add_space(6.0);

            // 2. Quick-action chips
            let mut chip_prompt: Option<(String, bool)> = None;
            let mut heal_blemishes = false;
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                for (label, prompt_text) in [
                    ("✨ Auto-Retouch", ""),
                    ("🧍 Body Slim", "Slim body, flatten stomach, cinch waist, slim arms, and lengthen legs"),
                    ("☀ Auto Tone", "Auto tone and balance exposure, highlights, and shadows"),
                    ("🔥 Warm Glow", "Warm golden hour skin tone and sunlit white balance"),
                    ("🎬 Cinematic", "Cinematic teal and orange color grade with subtle vignette"),
                    ("🌑 Darken BG", "Darken the background and make the subject pop"),
                    ("🦷 Eyes & Teeth", "Brighten eyes, whiten sclera and teeth, and lift smile"),
                    ("↩ Undo", "undo"),
                ] {
                    let chip_btn = egui::Button::new(
                        RichText::new(crate::i18n::tr(label))
                            .size(11.0)
                            .color(Color32::from_rgb(212, 220, 236)),
                    )
                    .fill(Color32::from_rgb(35, 39, 49))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(56, 63, 78)))
                    .corner_radius(CornerRadius::same(12));
                    if ui.add(chip_btn).clicked() {
                        chip_prompt = Some((prompt_text.to_string(), sel_n > 1));
                    }
                }
                let blemish_btn = egui::Button::new(
                    RichText::new(crate::i18n::tr("🩹 Heal Blemishes"))
                        .size(11.0)
                        .color(Color32::from_rgb(212, 220, 236)),
                )
                .fill(Color32::from_rgb(35, 39, 49))
                .stroke(Stroke::new(1.0, Color32::from_rgb(56, 63, 78)))
                .corner_radius(CornerRadius::same(12));
                if ui.add(blemish_btn).clicked() {
                    heal_blemishes = true;
                }
            });

            if heal_blemishes {
                let all_sel = sel_n > 1;
                app.ui.ai_messages.push(AiChatMessage {
                    role: "user".into(),
                    text: crate::i18n::tr("🩹 Heal facial blemishes").to_string(),
                });
                match app.run("spot.findBlemishes", json!({"allSelected": all_sel})) {
                    Ok(r) => {
                        let n = r["added"].as_u64().unwrap_or(0);
                        let reply = if n == 0 {
                            crate::i18n::tr("No new facial blemishes detected.").to_string()
                        } else {
                            format!("Healed {n} facial blemish{} using PatchMatch.", if n == 1 { "" } else { "es" })
                        };
                        app.ui.ai_summary = reply.clone();
                        app.ui.ai_messages.push(AiChatMessage { role: "assistant".into(), text: reply.clone() });
                        app.toast(ctx, reply);
                    }
                    Err(e) => {
                        app.ui.ai_messages.push(AiChatMessage { role: "assistant".into(), text: e.to_string() });
                        app.toast(ctx, e);
                    }
                }
            }

            if let Some((p, batch)) = chip_prompt {
                send_message(app, ctx, &p, batch);
            }

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(4.0);

            // 3. Scrollable chat message list
            egui::ScrollArea::vertical()
                .id_salt("ai_studio_chat_scroll")
                .max_height(156.0)
                .min_scrolled_height(136.0)
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if app.ui.ai_messages.is_empty() {
                        egui::Frame::NONE
                            .fill(Color32::from_rgb(30, 34, 44))
                            .corner_radius(CornerRadius::same(8))
                            .inner_margin(Margin::same(10))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(crate::i18n::tr(
                                        "Chat to edit anything in the photo:\n• \"Brighten exposure & lift shadows\"\n• \"Smooth skin, remove shine & fix dark circles\"\n• \"Darken the background & warm skin tone\"\n• \"Crop 4x5 and add a cinematic grade\"",
                                    ))
                                    .size(11.5)
                                    .color(Color32::from_rgb(184, 194, 212)),
                                );
                            });
                    } else {
                        for msg in &app.ui.ai_messages {
                            let is_user = msg.role == "user";
                            ui.add_space(3.0);
                            ui.horizontal(|ui| {
                                if is_user {
                                    ui.add_space(28.0);
                                }
                                let bg = if is_user {
                                    Color32::from_rgb(44, 88, 158)
                                } else {
                                    Color32::from_rgb(33, 37, 47)
                                };
                                let fg = if is_user {
                                    Color32::from_rgb(245, 248, 255)
                                } else {
                                    Color32::from_rgb(214, 224, 242)
                                };
                                egui::Frame::NONE
                                    .fill(bg)
                                    .corner_radius(CornerRadius::same(8))
                                    .inner_margin(Margin { left: 9, right: 9, top: 6, bottom: 6 })
                                    .show(ui, |ui| {
                                        ui.set_max_width(card_w - 64.0);
                                        ui.label(RichText::new(&msg.text).size(11.5).color(fg));
                                    });
                            });
                        }
                    }

                    if app.ui.ai_busy {
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new(crate::i18n::tr("✨ Refining with AI Vision…"))
                                .size(11.0)
                                .italics()
                                .color(t.accent),
                        );
                    }
                });

            ui.add_space(6.0);

            // 4. Chat input row
            let mut submit = false;
            ui.horizontal(|ui| {
                let input_w = (ui.available_width() - 56.0).max(100.0);
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut app.ui.ai_prompt)
                        .id(egui::Id::new("ai_studio_chat_input"))
                        .hint_text(crate::i18n::tr("Ask AI to change anything in the image…"))
                        .desired_width(input_w),
                );
                if std::mem::take(&mut app.ui.ai_chat_focus) {
                    resp.request_focus();
                }
                let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let send_btn = egui::Button::new(
                    RichText::new(crate::i18n::tr("Send"))
                        .size(11.5)
                        .strong()
                        .color(Color32::WHITE),
                )
                .fill(t.accent)
                .corner_radius(CornerRadius::same(6));
                if ui.add(send_btn).clicked() || enter {
                    submit = true;
                }
            });

            if submit {
                let msg = std::mem::take(&mut app.ui.ai_prompt);
                if !msg.trim().is_empty() {
                    send_message(app, ctx, &msg, sel_n > 1);
                    app.ui.ai_chat_focus = true;
                }
            }
        });
}
