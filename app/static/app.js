/* Global Face Heritage Platform — front-end */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const fmt = (v, d = 3) => (v === null || v === undefined || v === "") ? "—" : (typeof v === "number" ? v.toFixed(d) : esc(v));
  const pct = (v) => (v * 100).toFixed(1) + "%";
  let token = localStorage.getItem("gfp_token") || null, me = null, file = null, TAX = null;
  const AUS = ["AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU12","AU15","AU17","AU20","AU23","AU25","AU26"];

  async function api(path, opts = {}) {
    const headers = opts.headers || {};
    if (token) headers["Authorization"] = "Bearer " + token;
    if (opts.json) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
    const r = await fetch(path, { ...opts, headers });
    if (r.status === 401) { logout(); throw new Error("Session expired — please sign in again."); }
    if (!r.ok) { let d = r.statusText; try { const j = await r.json(); d = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {} throw new Error(d); }
    return (r.headers.get("content-type") || "").includes("json") ? r.json() : r;
  }
  async function download(path, filename) { const r = await api(path); const a = document.createElement("a"); a.href = URL.createObjectURL(await r.blob()); a.download = filename; a.click(); }
  const pretty = id => String(id).replace(/_/g, " ").replace(/^./, c => c.toUpperCase());
  const tlabel = (field, id) => { if (!id) return ""; const t = TAX ? (TAX[field] || []).find(x => (x.id || x) === id) : null; return t && t.label ? t.label : pretty(id); };

  function show(view) {
    $$(".view").forEach(v => v.classList.add("hidden")); $("#view-" + view).classList.remove("hidden");
    $$("#nav button[data-view]").forEach(b => b.classList.toggle("active", b.dataset.view === view));
    if (view === "objects") loadObjects(); if (view === "references") loadReferences(); if (view === "admin") { loadAdmin(); loadTaxonomyView(); }
  }
  $$("#nav button[data-view]").forEach(b => b.onclick = () => show(b.dataset.view));
  function logout() { token = null; me = null; localStorage.removeItem("gfp_token"); $("#nav").classList.add("hidden"); show("auth"); }
  $("#logout").onclick = logout;

  async function boot() {
    try { const h = await api("/api/health"); $("#ver").textContent = h.versions.platform; refBanner(h.reference_index); } catch {}
    if (!token) return show("auth");
    try { me = await api("/api/auth/me"); await onLogin(); } catch { logout(); }
  }
  async function onLogin() {
    TAX = await api("/api/taxonomy");
    fillTaxonomySelects();
    $("#nav").classList.remove("hidden"); $("#whoami").textContent = `${me.email} · ${me.role}`;
    $("#nav-admin").classList.toggle("hidden", me.role !== "admin");
    $$(".admin-only").forEach(e => e.classList.toggle("hidden", me.role !== "admin"));
    show("analyse");
  }
  function options(field, blank = "—") {
    const vals = TAX[field] || [];
    return `<option value="">${blank}</option>` + vals.map(v => { const id = v.id || v, lab = v.label || id; const extra = field === "culture" && v.region ? ` (${tlabel("region", v.region)})` : ""; return `<option value="${esc(id)}">${esc(lab)}${esc(extra)}</option>`; }).join("");
  }
  function fillTaxonomySelects() {
    $$("select[data-tax]").forEach(s => { s.innerHTML = options(s.dataset.tax); });
    $("#meta-form [name=domain]").value = "sculpture"; $("#meta-form [name=license]").value = "UNKNOWN";
    for (const sel of ["#obj-culture", "#coll-form [name=culture]"]) $(sel).innerHTML = `<option value="">${sel.includes("coll") ? "culture filter…" : "all cultures"}</option>` + TAX.culture.map(c => `<option value="${c.id}">${esc(c.label)}</option>`).join("");
    for (const sel of ["#obj-domain", "#coll-form [name=domain]"]) $(sel).innerHTML = `<option value="">${sel.includes("coll") ? "domain filter…" : "all domains"}</option>` + TAX.domain.map(c => `<option value="${c.id}">${esc(c.label)}</option>`).join("");
    // culture → region auto-fill
    $("#meta-form [name=culture]").onchange = e => { const t = TAX.culture.find(c => c.id === e.target.value); if (t && t.region) $("#meta-form [name=region]").value = t.region; };
  }
  function refBanner(ri) {
    const b = $("#ref-banner"); b.classList.remove("hidden");
    if (!ri.size) { b.classList.add("warn"); b.innerHTML = "<b>Reference corpus is empty</b> — retrieval, morphological affinity and suggested attribution need it. Administrator: open <b>Reference corpus → Build / extend corpus</b> (downloads open-access museum images; needs internet)."; }
    else b.innerHTML = `Reference index: <b>${ri.size}</b> specimens · ${esc(ri.dataset_version)}` + (/SYNTHETIC|demo/i.test(ri.dataset_version) ? ' <span class="pill warn">synthetic demo data</span>' : "");
  }

  // auth: forgot / reset
  function showAuthPanel(which) { for (const f of ["login-form", "register-form", "forgot-form", "reset-form"]) $("#" + f).classList.toggle("hidden", f !== which); $("#auth-msg").textContent = ""; }
  $("#forgot-link").onclick = e => { e.preventDefault(); showAuthPanel("forgot-form"); };
  $$(".back-login").forEach(a => a.onclick = e => { e.preventDefault(); showAuthPanel("login-form"); });
  $("#forgot-form").onsubmit = async e => { e.preventDefault(); const f = new FormData(e.target);
    try { const r = await api("/api/auth/forgot", { method: "POST", json: { email: f.get("email") } }); $("#auth-msg").className = "msg ok"; $("#auth-msg").textContent = r.message; } catch (err) { $("#auth-msg").className = "msg"; $("#auth-msg").textContent = err.message; } };
  const resetToken = new URLSearchParams(location.search).get("reset");
  if (resetToken) { showAuthPanel("reset-form"); }
  $("#reset-form").onsubmit = async e => { e.preventDefault(); const f = new FormData(e.target);
    try { await api("/api/auth/reset", { method: "POST", json: { token: resetToken, password: f.get("password") } }); history.replaceState(null, "", location.pathname); showAuthPanel("login-form"); $("#auth-msg").className = "msg ok"; $("#auth-msg").textContent = "Password changed — sign in with the new one."; } catch (err) { $("#auth-msg").className = "msg"; $("#auth-msg").textContent = err.message; } };

  // auth
  $$(".tab").forEach(t => t.onclick = () => { $$(".tab").forEach(x => x.classList.toggle("active", x === t)); $("#login-form").classList.toggle("hidden", t.dataset.tab !== "login"); $("#register-form").classList.toggle("hidden", t.dataset.tab !== "register"); });
  $("#login-form").onsubmit = async e => { e.preventDefault(); const f = new FormData(e.target);
    try { const r = await fetch("/api/auth/login", { method: "POST", body: new URLSearchParams({ username: f.get("email"), password: f.get("password") }) }); const j = await r.json(); if (!r.ok) throw new Error(j.detail || "Login failed");
      token = j.access_token; localStorage.setItem("gfp_token", token); me = j.user; await onLogin(); } catch (err) { $("#auth-msg").textContent = err.message; } };
  $("#register-form").onsubmit = async e => { e.preventDefault(); const f = new FormData(e.target);
    try { await api("/api/auth/register", { method: "POST", json: Object.fromEntries(f.entries()) }); $("#auth-msg").className = "msg ok"; $("#auth-msg").textContent = "Account created — sign in now."; $$(".tab")[0].click(); } catch (err) { $("#auth-msg").className = "msg"; $("#auth-msg").textContent = err.message; } };

  // upload
  const drop = $("#drop"), fileIn = $("#file");
  drop.onclick = () => fileIn.click(); drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); }; drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); };
  fileIn.onchange = () => fileIn.files[0] && setFile(fileIn.files[0]);
  function setFile(f) { file = f; faceBox = null; boxTool.classList.add("hidden"); const img = $("#preview"); img.src = URL.createObjectURL(f); img.classList.remove("hidden"); $("#drop-text").textContent = f.name; $("#run").disabled = false; }
  // ---------------------------------------------------------------- manual face rectangle (stylised / abstract faces)
  let faceBox = null;   // [x0,y0,x1,y1] in original pixels
  const boxTool = $("#boxtool"), boxCanvas = $("#boxcanvas"), boxOpen = $("#box-open");
  let boxImg = null, boxScale = 1, dragStart = null;
  function openBoxTool() {
    if (!file) return;
    boxImg = new Image();
    boxImg.onload = () => {
      const maxW = Math.min(boxTool.clientWidth || 600, 700);
      boxScale = Math.min(1, maxW / boxImg.naturalWidth, 520 / boxImg.naturalHeight);
      boxCanvas.width = Math.round(boxImg.naturalWidth * boxScale); boxCanvas.height = Math.round(boxImg.naturalHeight * boxScale);
      boxTool.classList.remove("hidden"); drawBox();
    };
    boxImg.src = URL.createObjectURL(file);
  }
  function drawBox(tmp) {
    if (!boxImg) return;
    const c = boxCanvas.getContext("2d"); c.drawImage(boxImg, 0, 0, boxCanvas.width, boxCanvas.height);
    const b = tmp || faceBox; if (!b) { $("#box-info").textContent = "no rectangle yet"; return; }
    c.lineWidth = 3; c.strokeStyle = "#e0552f"; c.strokeRect(b[0] * boxScale, b[1] * boxScale, (b[2] - b[0]) * boxScale, (b[3] - b[1]) * boxScale);
    $("#box-info").textContent = `face rectangle ${b[0]},${b[1]} → ${b[2]},${b[3]} px (used as geometric guidance only)`;
  }
  const evPos = e => { const r = boxCanvas.getBoundingClientRect(); const sx = boxCanvas.width / r.width, sy = boxCanvas.height / r.height; const p = e.touches ? e.touches[0] : e; return [Math.round((p.clientX - r.left) * sx / boxScale), Math.round((p.clientY - r.top) * sy / boxScale)]; };
  const onDown = e => { e.preventDefault(); dragStart = evPos(e); };
  const onMove = e => { if (!dragStart) return; e.preventDefault(); const p = evPos(e); drawBox([Math.min(dragStart[0], p[0]), Math.min(dragStart[1], p[1]), Math.max(dragStart[0], p[0]), Math.max(dragStart[1], p[1])]); };
  const onUp = e => { if (!dragStart) return; const p = evPos(e.changedTouches ? { clientX: e.changedTouches[0].clientX, clientY: e.changedTouches[0].clientY } : e); const b = [Math.min(dragStart[0], p[0]), Math.min(dragStart[1], p[1]), Math.max(dragStart[0], p[0]), Math.max(dragStart[1], p[1])]; dragStart = null; if (b[2] - b[0] > 8 && b[3] - b[1] > 8) faceBox = b; drawBox(); };
  boxCanvas.onmousedown = onDown; boxCanvas.onmousemove = onMove; boxCanvas.onmouseup = onUp; boxCanvas.onmouseleave = () => { dragStart = null; drawBox(); };
  boxCanvas.addEventListener("touchstart", onDown, { passive: false }); boxCanvas.addEventListener("touchmove", onMove, { passive: false }); boxCanvas.addEventListener("touchend", onUp);
  $("#box-clear").onclick = () => { faceBox = null; drawBox(); };
  boxOpen.onclick = e => { e.preventDefault(); openBoxTool(); };
  document.addEventListener("click", e => { if (e.target.closest(".open-box")) { show("analyse"); openBoxTool(); window.scrollTo(0, 0); } });
  $("#run").onclick = async () => {
    if (!file) return; $("#run").disabled = true; $("#run-msg").className = "msg"; $("#run-msg").innerHTML = '<span class="spinner"></span>Detecting faces, landmarks, facial actions, morphometrics, stability…';
    const all = Object.fromEntries([...new FormData($("#meta-form")).entries()].filter(([, v]) => String(v).trim() !== ""));
    const imgMeta = {}; for (const k of ["view_angle", "photographer"]) if (all[k]) { imgMeta[k] = all[k]; delete all[k]; }
    for (const k of ["date_start", "date_end"]) if (all[k]) all[k] = parseInt(all[k], 10);
    if (all.cultural_sensitivity) all.cultural_sensitivity = [all.cultural_sensitivity];
    const fd = new FormData(); fd.append("file", file); fd.append("metadata", JSON.stringify(all)); fd.append("image_metadata", JSON.stringify(imgMeta)); if (faceBox) fd.append("face_box", faceBox.join(","));
    try { const res = await api("/api/analyse", { method: "POST", body: fd }); $("#run-msg").textContent = ""; renderResult(res.report, res.image, [], res.duplicates); show("result"); }
    catch (err) { $("#run-msg").textContent = err.message; }
    $("#run").disabled = false;
  };

  // rendering helpers
  const pill = (t, c) => `<span class="pill ${c || ""}">${esc(t)}</span>`;
  const qcls = s => s >= 70 ? "ok" : s >= 40 ? "warn" : "bad";
  function bars(obj, hl) { return `<div class="bars">` + Object.entries(obj).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<span>${esc(k)}</span><div class="bar ${k === hl ? "" : "dim"}"><i style="width:${(v * 100).toFixed(1)}%"></i></div><span>${pct(v)}</span>`).join("") + `</div>`; }
  const kv = rows => `<div class="kv">` + rows.map(([k, v]) => `<b>${esc(k)}</b><span>${v}</span>`).join("") + `</div>`;
  function refCard(r) {
    return `<div class="ref"><img data-auth-src="/api/references/${encodeURIComponent(r.reference_id)}/image" alt=""><div class="b">
      <div class="t" title="${esc(r.title)}">${esc(r.reference_id)}</div><div>${esc(r.museum || "")}</div>
      <div>${esc(tlabel("culture", r.culture))} · ${esc(tlabel("region", r.region))}</div><div>${esc(r.date || "")} ${r.domain ? "· " + esc(tlabel("domain", r.domain)) : ""}</div>
      <div>${r.expression ? "Expr: <b>" + esc(r.expression) + "</b> " : ""}${/unverified/.test(r.expression_source || "") ? `<span class="pill" title="Expression proposed by the frozen model when the reference database was built; it becomes a validated label once an expert annotates it.">auto-labelled</span>` : ""}</div>
      <div>${r.similarity !== undefined ? "Similarity " + fmt(r.similarity) : ""}${r.procrustes_distance !== undefined ? "Procrustes d " + fmt(r.procrustes_distance, 4) : ""}</div>
      <div class="muted">${esc(r.license || "")}</div>${r.source_url ? `<a href="${esc(r.source_url)}" target="_blank" rel="noopener">source ↗</a>` : ""}</div></div>`;
  }
  async function hydrate(root) { for (const img of $$("img[data-auth-src]", root)) { try { const r = await api(img.dataset.authSrc); img.src = URL.createObjectURL(await r.blob()); } catch { img.alt = "n/a"; } } }

  function renderFace(f, imgId, rep) {
    if (f.status !== "completed") return `<div class="banner warn">Face #${f.face_index} rejected: ${esc(f.rejection_reason)}</div>`;
    const ex = f.expression, p = f.pose, q = f.quality, st = (f.stability || {}).summary || {}, m = f.morphometrics, rr = f.reference_retrieval;
    let h = `<div class="hero">
      <div class="stat"><div class="k">${rep.input.analysis_context.depicted ? "Depicted expression" : "Perceived expression"}</div><div class="v ${ex.is_unknown ? "unknown" : ""}">${esc(ex.primary)}</div><div class="k">confidence ${fmt(ex.confidence)}</div></div>
      <div class="stat"><div class="k">Face quality</div><div class="v">${q.score}</div><div class="k">${pill(qcls(q.score), qcls(q.score))} · difficulty level ${f.difficulty_level}</div></div>
      <div class="stat"><div class="k">Pose quality</div><div class="v">${fmt(p.pose_quality, 2)}</div><div class="k">yaw ${p.yaw_deg}° · pitch ${p.pitch_deg}° · roll ${p.roll_deg}°</div></div>
      <div class="stat"><div class="k">Stability</div><div class="v">${fmt(st.stability_score, 2)}</div><div class="k">${st.illumination_top_class_consistent === false ? pill("lighting-sensitive", "warn") : (st.stability_score === undefined ? "not run" : "illumination-consistent")}</div></div></div>
      <div class="imgs"><figure><img data-auth-src="/api/images/${imgId}/artefact/original"><figcaption>original image</figcaption></figure><figure><img data-auth-src="/api/images/${imgId}/artefact/${f.artefacts.crop}"><figcaption>face crop</figcaption></figure><figure><img data-auth-src="/api/images/${imgId}/artefact/${f.artefacts.aligned}"><figcaption>normalised &amp; aligned</figcaption></figure></div>
      <h3>Associated musculature of the detected facial actions</h3><div class="imgs"><figure><img class="wide-img" data-auth-src="/api/images/${imgId}/artefact/${f.artefacts.muscle_map || ""}"><figcaption>colour = AU, thickness = evidence; inferred from geometry — an artwork has no muscle movement</figcaption></figure></div>`;
    const sa = f.suggested_attribution;
    if (sa && Object.keys(sa.fields).length) {
      h += `<h3>Suggested attribution (from similarity to reference specimens)</h3><p class="muted small">${esc(sa.wording)}</p><table class="tbl"><thead><tr><th>Field</th><th>Suggested value</th><th>Confidence</th><th>Reliability</th><th>Alternatives</th><th>Supporting specimens</th></tr></thead><tbody>` +
        Object.entries(sa.fields).map(([k, v]) => `<tr><td>${esc(k)}</td><td><b>${esc(tlabel(k, v.value))}</b></td><td>${pct(v.confidence)}</td><td>${pill(v.reliability, v.reliability === "higher" ? "ok" : "warn")}</td><td class="small">${v.alternatives.map(a => `${esc(tlabel(k, a.value))} ${pct(a.share)}`).join(", ")}</td><td class="small">${v.supporting_specimens.map(esc).join(", ")}</td></tr>`).join("") +
        `</tbody></table><button class="btn" data-apply-suggestion='${esc(JSON.stringify(Object.fromEntries(Object.entries(sa.fields).map(([k, v]) => [k, v.value]))))}'>Apply suggested values to the object record</button> <span class="small muted">(only fills fields; you can edit them afterwards on the object page)</span>`;
    } else if (rr.reference_index_size === 0) {
      h += `<div class="banner warn">Suggested attribution (culture / region / period / material) needs the reference corpus. Ask the administrator to build it under <b>Reference corpus</b>.</div>`;
    }
    h += `<h3>Expression</h3><p class="muted small">${esc(rep.wording.expression)}</p>` + (ex.is_unknown ? `<div class="banner warn"><b>UNKNOWN / AMBIGUOUS:</b> ${esc(ex.unknown_reason)}</div>` : "") + bars(ex.probabilities, ex.is_unknown ? null : ex.primary);
    h += `<div class="research-only"><h3>Contributing regions</h3>${bars(ex.region_contributions)}<p class="small muted">${esc(ex.model)}</p>
      <h3>Facial-action configuration</h3><p class="muted small">${esc(rep.wording.actions)}</p><table class="tbl"><thead><tr><th>AU</th><th>Description</th><th>Intensity</th><th>Evidence</th><th>Associated musculature</th></tr></thead><tbody>` +
      f.facial_actions.action_units.map(u => `<tr style="${u.present ? "" : "opacity:.5"}"><td><b>${u.code}</b></td><td>${esc(u.name)}</td><td>${u.intensity}</td><td><div class="bar"><i style="width:${u.evidence * 100}%"></i></div></td><td>${esc(u.associated_musculature)}</td></tr>`).join("") + `</tbody></table></div>`;
    h += `<h3>Morphometrics <span class="pill">reference n = ${m.reference_sample_size}</span></h3><p>${esc(m.interpretation)}</p>
      <div class="imgs"><figure><img class="wide-img" data-auth-src="/api/images/${imgId}/artefact/${f.artefacts.morphometrics || ""}"><figcaption>(a) digitised landmarks and wireframe · (b) Procrustes superimposition with deviation vectors · (c) PCA with 95 % ellipses — as in Millones' analysis</figcaption></figure></div>`;
    for (const [fld, aff] of Object.entries(m.group_affinity || {})) h += `<h3 class="small">Affinity by ${esc(fld)} ${pill("reliability: " + aff.reliability, aff.reliability === "higher" ? "ok" : "warn")}</h3>` + bars(Object.fromEntries(aff.ranking.map(r => [`${tlabel(fld, r.group)} (n=${r.n})`, r.affinity])), `${tlabel(fld, aff.most_similar)} (n=${aff.ranking[0].n})`);
    h += `<div class="research-only">` + kv([["Procrustes distance to consensus", fmt(m.procrustes_distance_to_consensus, 4)], ["Ward cluster", m.cluster_id ?? "—"], ["PC scores", m.pca_scores.slice(0, 5).map(x => x.toFixed(4)).join(", ") || "—"], ["Explained variance", m.pca_explained_variance.slice(0, 5).map(pct).join(", ") || "—"]]) +
      `<h3 class="small">Linear indices</h3>` + kv(Object.entries(m.linear_measurements).map(([k, v]) => [k, fmt(v, 4)])) + `</div>` + m.caveats.map(c => `<div class="cav">${esc(c)}</div>`).join("");
    h += `<h3>Reference specimens — closest overall</h3>` + (rr.reference_index_size ? `<div class="ref-grid">${rr.nearest_overall.map(refCard).join("")}</div>` +
      (rr.nearest_same_expression.length ? `<h3>Reference specimens with the same expression (“${esc(ex.primary)}”)</h3><div class="ref-grid">${rr.nearest_same_expression.map(refCard).join("")}</div>` : `<p class="small muted">No reference specimen carries the label “${esc(ex.primary)}” yet.</p>`) +
      Object.entries(rr.nearest_same_group || {}).map(([fld, g]) => g.specimens.length ? `<h3>Reference specimens from the same ${esc(fld)} (${esc(tlabel(fld, g.value))})</h3><div class="ref-grid">${g.specimens.map(refCard).join("")}</div>` : "").join("")
      : `<div class="banner warn">Reference corpus empty — no comparison images available yet.</div>`);
    if (f.stability) { const il = f.stability.illumination, pt = f.stability.perturbation; h += `<div class="research-only"><h3>Stability</h3><div class="grid-2"><table class="tbl"><thead><tr><th>Normaliser</th><th>Top class</th></tr></thead><tbody>` + Object.entries(il).map(([k, r]) => `<tr><td>${esc(k)}</td><td>${esc(r.primary || r.error)}</td></tr>`).join("") + `</tbody></table><table class="tbl"><thead><tr><th>Perturbation</th><th>Top class</th><th>TV dist.</th></tr></thead><tbody>` + Object.entries(pt).map(([k, r]) => `<tr><td>${esc(k)}</td><td>${esc(r.primary || r.error)} ${r.same_top_class === false ? pill("changed", "warn") : ""}</td><td>${fmt(r.tv_distance, 4)}</td></tr>`).join("") + `</tbody></table></div></div>`; }
    return h;
  }

  function annotationForm(faceDbId) {
    return `<div class="card"><h2>Human annotation (gold-standard layer)</h2><p class="muted small">Stored with annotator, date, confidence and review status. Disagreements go to the admin queue; only accepted corrections enter the training export. Expert-layer annotations require a FACS-certified account.</p>
      <form class="fb" data-face="${faceDbId}">
        <label>Layer <select name="layer"><option value="gold_standard">gold standard</option><option value="expert">expert (FACS-certified)</option></select></label>
        <label>Verdict <select name="verdict"><option value="agree">Agree with model</option><option value="disagree">Disagree — correct</option><option value="uncertain">Uncertain</option></select></label>
        <label>Expression <select name="expression"><option value="">—</option>${["neutral","happiness","sadness","anger","fear","disgust","surprise","contempt","pain","determination_strain","unknown/ambiguous","other"].map(o => `<option>${o}</option>`).join("")}</select></label>
        <label>Confidence (0–1) <input name="confidence" type="number" min="0" max="1" step="0.05" value="0.8"></label>
        <label>Face completeness <select name="face_completeness">${options("face_completeness")}</select></label>
        <label>Representation <select name="representation_type">${options("representation_type")}</select></label>
        <label>Subject <select name="subject_type">${options("subject_type")}</select></label>
        <label>Difficulty (0–6) <input name="difficulty_level" type="number" min="0" max="6"></label>
        <div class="wide"><div class="small muted">Action units (present / absent / uncertain)</div><div class="au-grid">${AUS.map(a => `<label>${a}<select name="au_${a}"><option value="">—</option><option>present</option><option>absent</option><option>uncertain</option></select></label>`).join("")}</div></div>
        <label class="wide">Comment <textarea name="comment" rows="2"></textarea></label>
        <label><input type="checkbox" name="flag_for_review"> Flag for review</label><div><button class="primary" type="submit">Save annotation</button></div></form><p class="msg fb-msg"></p></div>`;
  }

  function renderResult(rep, image, annotations, dups) {
    const v = $("#view-result"), id = image.id, om = rep.object_metadata || {};
    let h = `<div class="card"><div class="row between"><h2>Image analysis <code>${esc(id)}</code> ${pill(rep.status, rep.status === "completed" ? "ok" : "bad")}</h2>
      <div class="dl"><button class="btn" data-dl="pdf">PDF</button><button class="btn" data-dl="json">JSON</button><button class="btn" data-dl="csv">CSV</button><button class="btn" data-rerun>Re-run</button><button class="btn" data-obj="${esc(om.id || "")}">Open object</button></div></div>
      <p class="muted small">${esc(image.original_filename)} · ${new Date(image.created_at).toLocaleString()} · SHA-256 ${esc(rep.input.sha256?.slice(0, 16))}… · pHash ${esc(rep.input.phash)} · ${esc(image.model_version)} · ${esc(image.dataset_version)}</p>
      <div class="fw">Object record (metadata layer — attached after analysis): <b>${esc(om.title || "untitled")}</b> · ${esc(tlabel("domain", om.domain))} · ${esc(tlabel("culture", om.culture))} · ${esc(tlabel("region", om.region))} · ${esc(tlabel("period", om.period))} · ${esc(tlabel("material", om.material))} · ${esc(om.date_display || "")} · licence ${esc(om.license)} ${om.training_allowed ? pill("training-eligible", "ok") : pill("not training-eligible", "warn")}</div>`;
    if (dups && (dups.exact || (dups.near || []).length)) h += `<div class="banner warn">Possible duplicate: ${dups.exact ? `identical file already exists (image ${esc(dups.exact)}). ` : ""}${(dups.near || []).map(n => `near-duplicate of image ${esc(n.image_id)} (object ${esc(n.object_id)}, Hamming ${n.hamming})`).join("; ")}</div>`;
    h += `<div class="imgs"><figure><img data-auth-src="/api/images/${id}/artefact/overlay.jpg"><figcaption>${rep.face_count} face(s) detected · landmarks</figcaption></figure></div>`;
    if (rep.warnings?.length) h += `<div class="warnlist">⚠ ${rep.warnings.map(esc).join(" · ")}</div>`;
    if (rep.status !== "completed") { h += `<div class="banner warn"><b>Not completed.</b> ${esc(rep.rejection_reason || rep.error || "")}${/face/i.test(rep.rejection_reason || "") ? ' <button class="btn open-box">Draw the face rectangle and run again</button>' : ""}</div></div>`; v.innerHTML = h; wire(v, id, image); hydrate(v); return; }
    h += `<div class="facetabs">${rep.faces.map(f => `<button data-face-tab="${f.face_index}" class="${f.face_index === 0 ? "active" : ""}">Face #${f.face_index} · ${esc(f.expression?.primary || f.status)}</button>`).join("")}</div></div>`;
    const dbFaces = image.faces_db || [];
    h += rep.faces.map(f => `<div class="card face-panel ${f.face_index === 0 ? "" : "hidden"}" data-face-panel="${f.face_index}">${renderFace(f, id, rep)}</div>` +
      `<div class="face-panel ${f.face_index === 0 ? "" : "hidden"}" data-face-panel="${f.face_index}">${annotationForm(f.__db_id || "")}` +
      (annotations.filter(a => a.face_index === f.face_index).length ? `<div class="card"><h2>Existing annotations</h2><table class="tbl"><thead><tr><th>Date</th><th>Layer</th><th>Verdict</th><th>Expression</th><th>Conf.</th><th>Comment</th><th>Review</th></tr></thead><tbody>${annotations.filter(a => a.face_index === f.face_index).map(a => `<tr><td>${new Date(a.created_at).toLocaleString()}</td><td>${esc(a.layer)}</td><td>${esc(a.verdict)}</td><td>${esc(a.expression || "")}</td><td>${fmt(a.confidence, 2)}</td><td>${esc(a.comment || "")}</td><td>${esc(a.review_status)}</td></tr>`).join("")}</tbody></table></div>` : "") + `</div>`).join("");
    h += `<div class="card research-only"><h2>Reproducibility</h2>${kv(Object.entries(rep.versions).map(([k, val]) => [k, esc(val)]))}<p class="small muted">Parameters <code>${esc(JSON.stringify(rep.parameters))}</code> · ${rep.duration_s}s</p></div>`;
    v.innerHTML = h; wire(v, id, image); hydrate(v);
  }
  function wire(v, id, image) {
    $$("[data-dl]", v).forEach(b => b.onclick = () => download(`/api/images/${id}/download?format=${b.dataset.dl}`, `face_heritage_${id.slice(0, 12)}.${b.dataset.dl}`));
    const rr = $("[data-rerun]", v); if (rr) rr.onclick = async () => { rr.disabled = true; try { const res = await api(`/api/images/${id}/rerun`, { method: "POST" }); openImage(res.image.id); } catch (e) { alert(e.message); } };
    const ob = $("[data-obj]", v); if (ob && ob.dataset.obj) ob.onclick = () => openObject(ob.dataset.obj);
    $$("[data-apply-suggestion]", v).forEach(b => b.onclick = async () => { try { await api(`/api/objects/${image.object_id}/apply-suggestion`, { method: "POST", json: JSON.parse(b.dataset.applySuggestion) }); b.textContent = "Applied ✓"; b.disabled = true; } catch (e) { alert(e.message); } });
    $$("[data-face-tab]", v).forEach(b => b.onclick = () => { $$("[data-face-tab]", v).forEach(x => x.classList.toggle("active", x === b)); $$("[data-face-panel]", v).forEach(p => p.classList.toggle("hidden", p.dataset.facePanel !== b.dataset.faceTab)); });
    $$("form.fb", v).forEach(f => f.onsubmit = async e => {
      e.preventDefault(); const fd = new FormData(f); const faceId = f.dataset.face; const msg = f.parentElement.querySelector(".fb-msg");
      if (!faceId) { msg.textContent = "Open the analysis from the Objects page to annotate (face record id needed)."; return; }
      const aus = {}; for (const a of AUS) if (fd.get("au_" + a)) aus[a] = fd.get("au_" + a);
      const body = { layer: fd.get("layer"), verdict: fd.get("verdict"), expression: fd.get("expression") || null, confidence: fd.get("confidence") ? parseFloat(fd.get("confidence")) : null,
        face_completeness: fd.get("face_completeness") || null, representation_type: fd.get("representation_type") || null, subject_type: fd.get("subject_type") || null,
        difficulty_level: fd.get("difficulty_level") ? parseInt(fd.get("difficulty_level"), 10) : null, aus: Object.keys(aus).length ? aus : null, comment: fd.get("comment") || null, flag_for_review: fd.get("flag_for_review") === "on" };
      try { await api(`/api/faces/${faceId}/annotations`, { method: "POST", json: body }); msg.className = "msg ok fb-msg"; msg.textContent = "Annotation saved."; openImage(id); } catch (err) { msg.className = "msg fb-msg"; msg.textContent = err.message; }
    });
  }
  async function openImage(id) {
    try { const r = await api(`/api/images/${id}`); r.report.faces.forEach(f => { const db = r.faces.find(x => x.face_index === f.face_index); f.__db_id = db ? db.id : ""; }); renderResult(r.report, r.image, r.annotations, null); show("result"); } catch (e) { alert(e.message); }
  }

  // objects
  async function loadObjects() {
    const p = new URLSearchParams({ q: $("#obj-q").value, culture: $("#obj-culture").value, domain: $("#obj-domain").value, mine: $("#obj-mine").checked });
    const rows = await api("/api/objects?" + p);
    $("#obj-table tbody").innerHTML = rows.map(o => `<tr><td>${new Date(o.created_at).toLocaleDateString()}</td><td><b>${esc(o.title || "untitled")}</b><div class="small muted">${esc(o.object_number || "")}</div></td><td>${esc(tlabel("domain", o.domain))}</td><td>${esc(tlabel("culture", o.culture))}<div class="small muted">${esc(tlabel("region", o.region))}</div></td><td>${esc(tlabel("period", o.period))}<div class="small muted">${esc(o.date_display || (o.date_start != null ? o.date_start + "–" + o.date_end : ""))}</div></td><td>${esc(o.institution || o.source)}</td><td>${esc(o.license)} ${o.training_allowed ? pill("train", "ok") : ""}</td><td>${o.image_count} / ${o.face_count}</td><td><button class="btn" data-open="${o.id}">Open</button></td></tr>`).join("") || `<tr><td colspan="9" class="muted">No objects.</td></tr>`;
    $$("#obj-table [data-open]").forEach(b => b.onclick = () => openObject(b.dataset.open));
  }
  ["#obj-q", "#obj-culture", "#obj-domain", "#obj-mine"].forEach(s => $(s).onchange = loadObjects); $("#obj-q").oninput = () => { clearTimeout(window._t); window._t = setTimeout(loadObjects, 300); };
  async function openObject(id) {
    const r = await api(`/api/objects/${id}`); const o = r.object, v = $("#view-object");
    v.innerHTML = `<div class="card"><div class="row between"><h2>${esc(o.title || "untitled")} <span class="pill">${esc(tlabel("domain", o.domain))}</span></h2><div class="row"><label class="btn">Add image <input type="file" id="obj-add-img" hidden accept=".jpg,.jpeg,.png,.tif,.tiff,.webp"></label><button class="btn" data-del-obj="${o.id}">Delete object</button></div></div>
      <div class="crumbs">${r.region_path.map(x => esc(tlabel("region", x))).join(" › ")} › ${esc(tlabel("culture", o.culture))} ${o.culture_subgroup ? "› " + esc(o.culture_subgroup) : ""}</div>
      ${kv(Object.entries(o).filter(([k, val]) => val !== null && val !== "" && !["id"].includes(k) && typeof val !== "object").map(([k, val]) => [k, esc(["domain","region","culture","period","material","technique","object_type","subject_type","representation_type","license"].includes(k) ? tlabel(k, val) : val)]))}
      ${r.collections.length ? `<p class="small">Collections: ${r.collections.map(c => pill(c.name + " v" + c.version)).join(" ")}</p>` : ""}</div>
      <div class="card"><h2>Images (${r.images.length}) and faces (${r.faces.length})</h2><table class="tbl"><thead><tr><th>Uploaded</th><th>File</th><th>Status</th><th>Quality</th><th>Faces</th><th>Expressions</th><th>Model</th><th></th></tr></thead><tbody>${r.images.map(im => `<tr><td>${new Date(im.created_at).toLocaleString()}</td><td>${esc(im.original_filename)}</td><td>${pill(im.status, im.status === "completed" ? "ok" : "bad")}</td><td>${im.quality_score ?? "—"}</td><td>${im.face_count}</td><td>${r.faces.filter(f => f.image_id === im.id).map(f => `#${f.face_index} ${esc(f.expression || f.status)}${f.flagged ? " ⚑" : ""}`).join("<br>")}</td><td class="small">${esc(im.model_version || "")}</td><td><button class="btn" data-img="${im.id}">Open</button></td></tr>`).join("")}</tbody></table></div>`;
    $$("[data-img]", v).forEach(b => b.onclick = () => openImage(b.dataset.img));
    $("[data-del-obj]", v).onclick = async () => { if (confirm("Delete object, its images and analyses?")) { await api(`/api/objects/${o.id}`, { method: "DELETE" }); show("objects"); } };
    $("#obj-add-img", v).onchange = async e => { const f = e.target.files[0]; if (!f) return; const fd = new FormData(); fd.append("file", f); fd.append("image_metadata", "{}"); try { const res = await api(`/api/objects/${o.id}/images`, { method: "POST", body: fd }); openImage(res.image.id); } catch (err) { alert(err.message); } };
    show("object");
  }

  // corpus builder (admin)
  let corpusTimer = null;
  async function pollCorpus() {
    try { const st = await api("/api/admin/corpus/status"); const box = $("#corpus-status");
      box.textContent = (st.running ? `RUNNING — ${st.current || ""}\n` : (st.name ? `finished: ${st.ok ? "OK" : "with errors"} · reference index ${st.reference_index.size} specimens (${st.reference_index.dataset_version})\n` : "")) + (st.log_tail || "");
      box.scrollTop = box.scrollHeight;
      if (!st.running) { clearTimeout(corpusTimer); corpusTimer = null; if (st.name) { refBanner(st.reference_index); loadReferences(); } } else corpusTimer = setTimeout(pollCorpus, 3000);
    } catch (e) { $("#corpus-status").textContent = e.message; }
  }
  async function loadKeys() { try { const k = await api("/api/admin/corpus/keys"); $("#keys-status").textContent = "saved: " + (Object.entries(k).filter(([, v]) => v).map(([n]) => n.replace("_API_KEY", "")).join(", ") || "none"); } catch { } }
  $("#keys-form").onsubmit = async e => { e.preventDefault(); const f = e.target; const body = {}; for (const el of f.querySelectorAll("input")) if (el.value.trim()) body[el.name] = el.value.trim(); try { const k = await api("/api/admin/corpus/keys", { method: "POST", json: body }); for (const el of f.querySelectorAll("input")) el.value = ""; $("#keys-status").textContent = "saved: " + (Object.entries(k).filter(([, v]) => v).map(([n]) => n.replace("_API_KEY", "")).join(", ") || "none"); } catch (err) { alert(err.message); } };
  $("#corpus-form").onsubmit = async e => {
    e.preventDefault(); const f = e.target;
    const sources = [...f.sources.selectedOptions].map(o => o.value); const queries = f.queries.value.split("\n").map(x => x.trim()).filter(Boolean);
    try { await api("/api/admin/corpus/build", { method: "POST", json: { sources, queries: queries.length ? queries : null, limit: parseInt(f.limit.value, 10) } }); $("#corpus-status").textContent = "starting…"; pollCorpus(); }
    catch (err) { alert(err.message); }
  };

  // references
  function statBoxes(obj) { return Object.entries(obj).map(([k, v]) => `<div class="stat"><div class="v">${(v && typeof v === "object") ? Object.values(v).reduce((a, b) => a + b, 0) : esc(v ?? "—")}</div><div class="k">${esc(k)}${(v && typeof v === "object") ? `<div>${Object.entries(v).slice(0, 12).map(([a, b]) => `${esc(a)}: ${b}`).join(" · ")}</div>` : ""}</div></div>`).join(""); }
  async function loadReferences() {
    if (me && me.role === "admin" && !corpusTimer) { pollCorpus(); loadKeys(); }
    const s = await api("/api/references/stats"); $("#ref-version").textContent = s.dataset_version; $("#ref-persist").textContent = s.size ? `${s.size} specimens kept on disk (built ${s.built_at ? new Date(s.built_at).toLocaleString() : "—"}). The database is persistent: it is loaded automatically at every start and only needs rebuilding to add new sources.` : "No reference database yet — the administrator builds it once from the panel above; it is then kept on disk permanently.";
    $("#ref-stats").innerHTML = statBoxes({ specimens: s.size, "by domain": s.by_domain, "by culture": s.by_culture, "by region": s.by_region, "by period": s.by_period, "by licence": s.by_license, "by expression label": s.by_expression });
    const r = await api("/api/references?limit=300"); $("#ref-grid").innerHTML = r.items.map(x => refCard({ ...x, reference_id: x.id, museum: x.institution, date: x.date_display })).join("") || `<p class="muted">Empty.</p>`; hydrate($("#ref-grid"));
  }
  async function loadTaxonomyView() {
    const st = await api("/api/taxonomy/stats");
    $("#tax-stats").innerHTML = statBoxes({ objects: st.objects, images: st.images, faces: st.faces, "objects by domain": st.domain, "objects by culture": st.culture, "objects by region": st.region });
    $("#tax-body").innerHTML = ["domain", "region", "culture", "period", "material", "technique", "object_type", "subject_type", "representation_type", "license", "cultural_sensitivity"].map(f => `<details><summary>${f} (${TAX[f].length})</summary><table class="tbl small"><tbody>${TAX[f].map(t => typeof t === "string" ? `<tr><td>${t}</td></tr>` : `<tr><td><code>${esc(t.id)}</code></td><td>${esc(t.label)}</td><td>${esc(t.region ? tlabel("region", t.region) : (t.parent ? "↳ " + tlabel("region", t.parent) : ""))}</td><td>${t.date_start !== undefined ? esc(t.date_start + " – " + t.date_end) : ""}</td><td class="muted">${esc(t.training_allowed !== undefined ? "training: " + t.training_allowed : (t.external_ids ? JSON.stringify(t.external_ids) : ""))}</td></tr>`).join("")}</tbody></table></details>`).join("");
  }

  // admin
  async function loadAdmin() {
    const s = await api("/api/admin/stats");
    $("#admin-stats").innerHTML = statBoxes({ users: s.users, objects: s.objects, images: s.images, faces: s.faces, annotations: s.annotations, "by layer": s.annotations_by_layer, verdicts: s.verdicts, "human–model agreement": s.human_model_agreement, "open flags": s.open_flags, "training-eligible objects": s.training_eligible_objects, "faces by expression": s.faces_by_expression, "objects by domain": s.objects_by_domain, "reference specimens": s.reference_index.size });
    const colls = await api("/api/admin/collections");
    $("#coll-table tbody").innerHTML = colls.map(c => `<tr><td>${esc(c.name)}</td><td>${esc(c.version)}</td><td class="small">${esc(JSON.stringify(c.filter || {}))}</td><td>${c.n_objects}</td><td><button class="btn" data-refresh="${c.id}">Refresh (new version)</button></td></tr>`).join("") || `<tr><td colspan="5" class="muted">No collections yet — e.g. create “Moche portrait vessels” with culture = Moche.</td></tr>`;
    $$("[data-refresh]").forEach(b => b.onclick = async () => { await api(`/api/admin/collections/${b.dataset.refresh}/refresh`, { method: "POST" }); loadAdmin(); });
    const flags = await api("/api/admin/flags?status=open");
    $("#flags-table tbody").innerHTML = flags.map(f => `<tr><td>${new Date(f.created_at).toLocaleString()}</td><td><a href="#" data-openimg="${f.image_id}">${esc(f.object_title || f.object_id)}</a> · face #${f.face_index}<div class="small muted">${esc(tlabel("culture", f.culture))}</div></td><td>${esc(f.model_expression)} (${fmt(f.model_confidence, 2)})</td><td>${esc(f.verdict)} → <b>${esc(f.expression || "")}</b> (conf ${fmt(f.confidence, 2)})</td><td>${esc(f.layer)}</td><td>${esc(f.comment || "")}</td><td><button class="btn" data-flag="${f.id}" data-dec="accepted">Accept</button> <button class="btn" data-flag="${f.id}" data-dec="rejected">Reject</button></td></tr>`).join("") || `<tr><td colspan="7" class="muted">No open flags.</td></tr>`;
    $$("[data-openimg]").forEach(a => a.onclick = e => { e.preventDefault(); openImage(a.dataset.openimg); });
    $$("[data-flag]").forEach(b => b.onclick = async () => { await api(`/api/admin/flags/${b.dataset.flag}`, { method: "PATCH", json: { review_status: b.dataset.dec } }); loadAdmin(); });
    const users = await api("/api/admin/users");
    $("#users-table tbody").innerHTML = users.map(u => `<tr><td>${u.id}</td><td>${esc(u.email)}</td><td>${esc(u.full_name)}</td><td>${esc(u.institution)}</td><td><select data-role="${u.id}">${["researcher", "annotator", "admin"].map(r => `<option ${u.role === r ? "selected" : ""}>${r}</option>`).join("")}</select></td><td><input type="checkbox" data-facs="${u.id}" ${u.facs_certified ? "checked" : ""}></td><td><input type="checkbox" data-active="${u.id}" ${u.is_active ? "checked" : ""}></td><td>${u.last_login ? new Date(u.last_login).toLocaleString() : "—"}</td><td>${u.images_count}</td><td>${u.annotations_count}</td><td>${u.email_verified ? "" : `${pill("not confirmed", "warn")} <button class="btn" data-verify="${u.id}">Confirm</button> `}${u.reset_requested_at ? pill("reset requested " + new Date(u.reset_requested_at).toLocaleDateString(), "warn") + " " : ""}<button class="btn" data-resetlink="${u.id}">Reset link</button></td></tr>`).join("");
    $$("[data-verify]").forEach(b => b.onclick = async () => { await api(`/api/admin/users/${b.dataset.verify}/verify`, { method: "POST" }); loadAdmin(); });
    $$("[data-resetlink]").forEach(b => b.onclick = async () => { const r = await api(`/api/admin/users/${b.dataset.resetlink}/reset-link`, { method: "POST" }); prompt("Give this link to the user (valid 24 h):", r.link); });
    try { const ms = await api("/api/admin/mail-status"); $("#mail-status").textContent = ms.configured ? `E-mail service configured — confirmation and reset e-mails are sent automatically (links use ${ms.public_url}).` : "No e-mail service configured: accounts are active immediately and password-reset requests appear here for you to hand over. To enable e-mails set the SMTP settings (see docs/DEPLOY.md)."; } catch {}
    const upd = (id, row) => api(`/api/admin/users/${id}`, { method: "PATCH", json: { role: $("select", row).value, is_active: $("[data-active]", row).checked, facs_certified: $("[data-facs]", row).checked } }).catch(e => alert(e.message));
    $$("[data-role],[data-facs],[data-active]").forEach(el => el.onchange = () => upd(el.dataset.role || el.dataset.facs || el.dataset.active, el.closest("tr")));
    const audit = await api("/api/admin/audit?limit=300");
    $("#audit-table tbody").innerHTML = audit.map(r => `<tr><td>${new Date(r.created_at).toLocaleString()}</td><td>${r.user_id ?? ""}</td><td>${esc(r.action)}</td><td>${esc(r.target || "")}</td><td>${esc(JSON.stringify(r.details || {})).slice(0, 120)}</td><td>${esc(r.ip || "")}</td></tr>`).join("");
  }
  $("#coll-form").onsubmit = async e => { e.preventDefault(); const f = new FormData(e.target); const filter = {}; if (f.get("culture")) filter.culture = f.get("culture"); if (f.get("domain")) filter.domain = f.get("domain"); await api("/api/admin/collections", { method: "POST", json: { name: f.get("name"), version: f.get("version"), filter_json: filter } }); e.target.reset(); loadAdmin(); };
  $("#export-gold").onclick = () => download("/api/admin/export/gold", "gold_standard.jsonl");
  $("#scan-dups").onclick = async () => { const r = await api("/api/admin/duplicates/scan", { method: "POST" }); alert(`Duplicate scan: ${r.pairs_linked} pairs linked, ${r.groups_created} groups created (review in Objects).`); };
  $("#reload-ref").onclick = async () => { const r = await api("/api/admin/references/reload", { method: "POST" }); refBanner(r); alert(`Reference index reloaded: ${r.size}`); };
  $("#reload-tax").onclick = async () => { await api("/api/admin/taxonomy/reload", { method: "POST" }); TAX = await api("/api/taxonomy"); fillTaxonomySelects(); alert("Taxonomy reloaded"); };
  $("#plan-split").onclick = async () => { const r = await api(`/api/admin/splits/plan?level=${$("#split-level").value}`); $("#split-out").textContent = JSON.stringify(r, null, 2); };

  boot();
})();
