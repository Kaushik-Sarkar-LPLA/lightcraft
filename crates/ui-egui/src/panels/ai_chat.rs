//! Sidebar AI Studio Chat panel (`RightPanel::AiChat`): persistent conversational control over
//! every setting in the image (Light, Color, White Balance, Grading, Mixer, Effects, Detail,
//! Crop, Masks, Blemish Healing, and all 34 Portrait & Body Retouch sliders).

use egui::{Color32, CornerRadius, Margin, RichText, Sense, Stroke, vec2};
use serde_json::json;

use crate::LightcraftApp;
use crate::state::AiChatMessage;
use crate::theme::Tokens;
use crate::widgets::{divider, register};

/// Format a model ID into a human-friendly display label.
pub fn format_model_label(model_id: &str, available: &[(String, String)]) -> String {
    if let Some((_, label)) = available.iter().find(|(id, _)| id == model_id) {
        return label.clone();
    }
    match model_id {
        "gemini-4-argon-high" => "Gemini 4 Argon (High)".into(),
        "gemini-4-argon-medium" => "Gemini 4 Argon (Medium)".into(),
        "gemini-4-argon-low" => "Gemini 4 Argon (Fast)".into(),
        "gemini-3.8-flash-high" => "Gemini 3.8 Flash (High)".into(),
        "gemini-3.8-flash-medium" => "Gemini 3.8 Flash (Medium)".into(),
        "gemini-3.8-flash-low" => "Gemini 3.8 Flash (Fast)".into(),
        "claude-opus-5-5-high" => "Claude Opus 5.5 (High)".into(),
        "claude-opus-5-5-medium" => "Claude Opus 5.5 (Medium)".into(),
        "claude-opus-5-5-low" => "Claude Opus 5.5 (Low)".into(),
        "claude-sonnet-5-5-high" => "Claude Sonnet 5.5 (High)".into(),
        "claude-sonnet-5-5-medium" => "Claude Sonnet 5.5 (Medium)".into(),
        "claude-sonnet-5-5-low" => "Claude Sonnet 5.5 (Low)".into(),
        "gpt-oss-120b-medium" => "GPT-OSS 120B".into(),
        other => other.to_string(),
    }
}

/// Trigger an AI image edit / chat turn from either the AI Chat sidebar or the Retouch panel.
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
                blemish_note = format!("Healed {n} blemish{}.", if n == 1 { "" } else { "es" });
            }
        }
    }

    if cfg!(target_arch = "wasm32") {
        // On Web, dispatch every chat message directly to Gemini Vision so the assistant responds
        // intelligently without pushing a synthetic local placeholder message.
        let telemetry = app.run("retouch.analyze", json!({})).unwrap_or_else(|_| json!({}));
        if clean_prompt.is_empty() {
            let _ = app.run("retouch.aiApply", json!({"prompt": "", "allSelected": all_selected}));
        }
        let history: Vec<serde_json::Value> = app
            .ui
            .ai_messages
            .iter()
            .rev()
            .take(10)
            .rev()
            .map(|m| json!({"role": m.role, "content": m.text}))
            .collect();
        app.ui.ai_busy = true;
        app.ui.ai_pending_request = Some(json!({
            "endpoint": app.ui.ai_endpoint.trim(),
            "model": app.ui.ai_model.trim(),
            "prompt": clean_prompt,
            "allSelected": all_selected,
            "blemishNote": blemish_note,
            "history": history,
            "telemetry": telemetry,
        }));
    } else {
        // Offline / native desktop execution path:
        match app.run("retouch.aiApply", json!({"prompt": clean_prompt, "allSelected": all_selected})) {
            Ok(r) => {
                let mut summary = r["summary"].as_str().unwrap_or("Updated image settings.").to_string();
                if !blemish_note.is_empty() {
                    summary = format!("{summary} {blemish_note}");
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
            }
        }
    }
}

/// Render the AI Studio Chat inside the right-hand sidebar (`RightPanel::AiChat`).
pub fn show_panel(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let ctx = ui.ctx().clone();
    let t = Tokens::get(&ctx);
    let sel_n = app.session.selection.ids.len();

    if app.ui.ai_available_models.is_empty() {
        app.ui.ai_available_models = crate::state::default_ai_models();
    }

    // 1. Sidebar Header + Model Selector
    egui::Frame::NONE
        .inner_margin(Margin { left: 20, right: 18, top: 14, bottom: 8 })
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let dot_color = if app.ui.ai_busy {
                    t.accent
                } else {
                    Color32::from_rgb(72, 208, 136)
                };
                let (dot_rect, _) = ui.allocate_exact_size(vec2(8.0, 8.0), Sense::hover());
                ui.painter().circle_filled(dot_rect.center(), 4.0, dot_color);

                ui.label(
                    RichText::new(crate::i18n::tr("AI Studio Chat"))
                        .font(t.semibold(15.0))
                        .color(t.text),
                );

                if sel_n > 1 {
                    ui.label(
                        RichText::new(format!("Batch ({sel_n})"))
                            .font(t.font(11.5))
                            .color(t.accent),
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if !app.ui.ai_messages.is_empty()
                        && ui
                            .add(egui::Button::new(RichText::new(crate::i18n::tr("Clear")).font(t.font(11.5)).color(t.text_dim)).frame(false))
                            .on_hover_text(crate::i18n::tr("Clear chat history"))
                            .clicked()
                    {
                        app.ui.ai_messages.clear();
                    }
                });
            });

            let active_label = if sel_n > 1 {
                format!("{sel_n} {}", crate::i18n::tr("photos selected for batch AI editing"))
            } else if let Some(id) = app.session.active()
                && let Some(p) = app.session.catalog.photo(id)
            {
                format!("{}: {}", crate::i18n::tr("Active"), p.file_name)
            } else {
                crate::i18n::tr("Select or import a photo to edit").to_string()
            };
            ui.add_space(2.0);
            ui.label(RichText::new(active_label).font(t.font(11.5)).color(t.text_dim));

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(crate::i18n::tr("Model")).font(t.font(11.5)).color(t.text_label));
                let current_label = format_model_label(&app.ui.ai_model, &app.ui.ai_available_models);
                let combo_w = (ui.available_width() - 4.0).max(120.0);
                let combo_resp = egui::ComboBox::from_id_salt("ai_chat_model_select")
                    .selected_text(RichText::new(current_label).font(t.font(11.5)).color(t.text))
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        let models = app.ui.ai_available_models.clone();
                        for (id, label) in models {
                            ui.selectable_value(&mut app.ui.ai_model, id, label);
                        }
                    });
                register(&ctx, "field:aiModelSelect", combo_resp.response.rect);
            });
        });

    divider(ui);

    // 2. Quick-action chips
    let mut chip_prompt: Option<(String, bool)> = None;
    let mut heal_blemishes = false;
    egui::Frame::NONE
        .inner_margin(Margin { left: 20, right: 18, top: 10, bottom: 10 })
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(5.0, 5.0);
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
                            .font(t.font(11.5))
                            .color(t.text_label),
                    )
                    .fill(t.button)
                    .stroke(Stroke::new(1.0, t.button_border))
                    .corner_radius(CornerRadius::same(12));
                    if ui.add(chip_btn).clicked() {
                        chip_prompt = Some((prompt_text.to_string(), sel_n > 1));
                    }
                }
                let blemish_btn = egui::Button::new(
                    RichText::new(crate::i18n::tr("🩹 Heal Blemishes"))
                        .font(t.font(11.5))
                        .color(t.text_label),
                )
                .fill(t.button)
                .stroke(Stroke::new(1.0, t.button_border))
                .corner_radius(CornerRadius::same(12));
                if ui.add(blemish_btn).clicked() {
                    heal_blemishes = true;
                }
            });
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
                app.toast(&ctx, reply);
            }
            Err(e) => {
                app.ui.ai_messages.push(AiChatMessage { role: "assistant".into(), text: e.to_string() });
                app.toast(&ctx, e);
            }
        }
    }

    if let Some((p, batch)) = chip_prompt {
        send_message(app, &ctx, &p, batch);
    }

    divider(ui);

    // 3. Scrollable chat history filling the sidebar above the bottom input bar
    let footer_h = 62.0;
    let scroll_h = (ui.available_height() - footer_h).max(120.0);
    let bubble_max_w = (ui.available_width() - 56.0).max(140.0);

    egui::ScrollArea::vertical()
        .id_salt("ai_studio_chat_scroll")
        .max_height(scroll_h)
        .min_scrolled_height(scroll_h)
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            egui::Frame::NONE
                .inner_margin(Margin { left: 18, right: 18, top: 10, bottom: 10 })
                .show(ui, |ui| {
                    if app.ui.ai_messages.is_empty() {
                        egui::Frame::NONE
                            .fill(t.button)
                            .stroke(Stroke::new(1.0, t.button_border))
                            .corner_radius(CornerRadius::same(8))
                            .inner_margin(Margin::same(12))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(crate::i18n::tr(
                                        "Chat to edit anything in the photo:\n\n• \"Brighten exposure & lift shadows\"\n• \"Smooth skin, remove shine & fix dark circles\"\n• \"Flatten stomach, slim waist & arms\"\n• \"Darken the background & warm skin tone\"\n• \"Crop 4x5 and add a cinematic grade\"",
                                    ))
                                    .font(t.font(12.0))
                                    .color(t.text_dim),
                                );
                            });
                    } else {
                        for msg in &app.ui.ai_messages {
                            let is_user = msg.role == "user";
                            ui.add_space(6.0);
                            if is_user {
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                                    egui::Frame::NONE
                                        .fill(t.accent)
                                        .corner_radius(CornerRadius::same(10))
                                        .inner_margin(Margin { left: 11, right: 11, top: 7, bottom: 7 })
                                        .show(ui, |ui| {
                                            ui.set_max_width(bubble_max_w);
                                            ui.label(RichText::new(&msg.text).font(t.font(12.0)).color(Color32::WHITE));
                                        });
                                });
                            } else {
                                ui.horizontal(|ui| {
                                    egui::Frame::NONE
                                        .fill(t.button)
                                        .stroke(Stroke::new(1.0, t.button_border))
                                        .corner_radius(CornerRadius::same(10))
                                        .inner_margin(Margin { left: 11, right: 11, top: 7, bottom: 7 })
                                        .show(ui, |ui| {
                                            ui.set_max_width(bubble_max_w);
                                            ui.label(RichText::new(&msg.text).font(t.font(12.0)).color(t.text));
                                        });
                                });
                            }
                        }
                    }

                    if app.ui.ai_busy {
                        ui.add_space(8.0);
                        let m_label = format_model_label(&app.ui.ai_model, &app.ui.ai_available_models);
                        ui.label(
                            RichText::new(format!("✨ {} {m_label}…", crate::i18n::tr("Thinking with")))
                                .font(t.font(11.5))
                                .italics()
                                .color(t.accent),
                        );
                    }
                });
        });

    divider(ui);

    // 4. Bottom-pinned chat input bar
    let mut submit = false;
    egui::Frame::NONE
        .inner_margin(Margin { left: 18, right: 18, top: 10, bottom: 10 })
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let input_w = (ui.available_width() - 60.0).max(90.0);
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut app.ui.ai_prompt)
                        .id(egui::Id::new("ai_studio_chat_input"))
                        .hint_text(crate::i18n::tr("Ask AI to change anything…"))
                        .desired_width(input_w),
                );
                register(&ctx, "field:aiChatInput", resp.rect);
                if std::mem::take(&mut app.ui.ai_chat_focus) {
                    resp.request_focus();
                }
                let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let send_btn = egui::Button::new(
                    RichText::new(crate::i18n::tr("Send"))
                        .font(t.semibold(12.0))
                        .color(Color32::WHITE),
                )
                .fill(t.accent)
                .corner_radius(CornerRadius::same(6));
                let send_resp = ui.add(send_btn);
                register(&ctx, "button:aiChatSend", send_resp.rect);
                if send_resp.clicked() || enter {
                    submit = true;
                }
            });
        });

    if submit {
        let msg = std::mem::take(&mut app.ui.ai_prompt);
        if !msg.trim().is_empty() {
            send_message(app, &ctx, &msg, sel_n > 1);
            app.ui.ai_chat_focus = true;
        }
    }
}
