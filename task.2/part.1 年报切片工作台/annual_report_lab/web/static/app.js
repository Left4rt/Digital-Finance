/* 年报采集与章节切片工作台 —— 前端逻辑 */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const KEYS = window.SECTION_KEYS || [];
  const NAMES = window.SECTION_NAMES || {};
  const LABEL = window.STATUS_LABEL || {};

  const BAD = ["NO_ANN", "DOWNLOAD_FAIL", "PDF_BROKEN", "NO_TEXT", "NO_SECTION", "ERROR"];
  const WARN = ["PARTIAL", "STOPPED"];

  let logOffset = 0;
  let timer = null;
  let lastRecords = [];
  let current = null;   // 抽屉当前记录

  /* ---------------- 参数自动保存 ---------------- */
  const SETTINGS_KEY = "annual_report_lab.settings.v1";

  function saveSettings() {
    try {
      localStorage.setItem(
        SETTINGS_KEY,
        JSON.stringify(collectConfig())
      );
    } catch (e) {
      console.warn("保存参数失败：", e);
    }
  }

  function restoreSettings() {
    let cfg = null;

    try {
      cfg = JSON.parse(
        localStorage.getItem(SETTINGS_KEY) || "null"
      );
    } catch (e) {
      console.warn("读取参数失败：", e);
      return;
    }

    if (!cfg) return;

    const values = {
      "#csv-path": cfg.csv_path,
      "#output": cfg.output,
      "#workers": cfg.workers,
      "#prefer": cfg.prefer,
      "#token": cfg.token,
      "#deepseek-key": cfg.deepseek_key,
    };

    Object.entries(values).forEach(([selector, value]) => {
      const el = $(selector);
      if (el && value !== undefined && value !== null) {
        el.value = value;
      }
    });

    // 同时兼容 select 和手动输入框
    const modelEl = $("#deepseek-model");
    if (modelEl && cfg.deepseek_model) {
      const model = String(cfg.deepseek_model);

      if (
        modelEl.tagName === "SELECT" &&
        !Array.from(modelEl.options).some(
          (option) => option.value === model
        )
      ) {
        modelEl.add(new Option(model, model));
      }

      modelEl.value = model;
    }

    const advEl = $("#deepseek-advanced-model");
    if (advEl && cfg.deepseek_advanced_model) {
      const m = String(cfg.deepseek_advanced_model);
      if (
        advEl.tagName === "SELECT" &&
        !Array.from(advEl.options).some((o) => o.value === m)
      ) {
        advEl.add(new Option(m, m));
      }
      advEl.value = m;
    }

    const checks = {
      "#overwrite": cfg.overwrite,
      "#resume": cfg.resume,
      "#save-fulltext": cfg.save_fulltext,
      "#ai-enabled": cfg.ai_enabled,
      "#ai-summary": cfg.ai_summary,
      "#ai-verify": cfg.ai_verify,
    };

    Object.entries(checks).forEach(([selector, checked]) => {
      const el = $(selector);
      if (el && typeof checked === "boolean") {
        el.checked = checked;
      }
    });

    if (Array.isArray(cfg.years)) {
      const selectedYears = new Set(
        cfg.years.map(String)
      );

      document
        .querySelectorAll("#years input")
        .forEach((input) => {
          input.checked = selectedYears.has(input.value);
        });
    }
  }

  function bindSettingsPersistence() {
    const selectors = [
      "#csv-path",
      "#output",
      "#workers",
      "#prefer",
      "#token",
      "#overwrite",
      "#resume",
      "#save-fulltext",
      "#ai-enabled",
      "#deepseek-key",
      "#deepseek-model",
      "#deepseek-advanced-model",
      "#ai-summary",
      "#ai-verify",
      "#years input",
    ].join(",");

    document
      .querySelectorAll(selectors)
      .forEach((el) => {
        el.addEventListener("input", saveSettings);
        el.addEventListener("change", saveSettings);
      });

    window.addEventListener("beforeunload", saveSettings);
  }

  /* ---------------- 工具 ---------------- */
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function tone(status) {
    if (status === "OK" || status === "SKIPPED") return "ok";
    if (BAD.includes(status)) return "bad";
    if (WARN.includes(status)) return "warn";
    if (status === "RUNNING") return "run";
    return "idle";
  }
  async function api(url, opts) {
    const r = await fetch(url, opts);
    let j = null;
    try { j = await r.json(); } catch (e) { j = { ok: false, msg: "服务返回异常" }; }
    return { status: r.status, body: j };
  }
  function post(url, data) {
    return api(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data || {}),
    });
  }

  /* ---------------- 清单校验 ---------------- */
  $("#btn-validate").addEventListener("click", async () => {
    const path = $("#csv-path").value.trim();
    if (!path) { $("#csv-info").textContent = "请先填写 CSV 路径"; return; }
    $("#csv-info").textContent = "校验中…";
    const { body } = await post("/api/validate", { csv_path: path });
    const box = $("#csv-preview");
    if (body.ok) {
      $("#csv-info").innerHTML =
        `编码 <b>${esc(body.encoding)}</b>，识别到 <b>${body.count}</b> 家公司`;
      box.classList.remove("hidden");
      box.innerHTML = body.preview
        .map((r) => `<div>${esc(r.ts_code)}　${esc(r.name || "—")}</div>`)
        .join("") + (body.count > 12 ? `<div>…共 ${body.count} 行</div>` : "");
    } else {
      $("#csv-info").textContent = "清单无法解析";
      box.classList.remove("hidden");
      box.innerHTML = (body.warnings || ["未知错误"])
        .map((w) => `<div>⚠ ${esc(w)}</div>`).join("");
    }
    if ((body.warnings || []).length && body.ok) {
      box.innerHTML += body.warnings.map((w) => `<div>⚠ ${esc(w)}</div>`).join("");
    }
  });

  /* ---------------- 网关检测 ---------------- */
  $("#btn-ping").addEventListener("click", async () => {
    setConn(null, "检测中…");
    const { body } = await post("/api/ping", { token: $("#token").value.trim() });
    setConn(body.ok, body.msg + (body.ok ? `（${body.count} 只在册）` : ""));
  });
  function setConn(ok, msg) {
    const p = $("#conn-pill");
    p.className = "pill " + (ok === null ? "idle" : ok ? "ok" : "bad");
    p.querySelector("span").textContent = msg;
  }

  /* ---------------- DeepSeek 检测 ---------------- */
  $("#btn-ping-ai").addEventListener("click", async () => {
    setAiConn(null, "检测中…");
    const { body } = await post("/api/ping_ai", {
      deepseek_key: $("#deepseek-key").value.trim(),
      deepseek_model: $("#deepseek-model").value,
      deepseek_advanced_model: $("#deepseek-advanced-model").value,
    });
    setAiConn(body.ok, body.msg);
  });
  function setAiConn(ok, msg) {
    const p = $("#ai-pill");
    p.className = "pill " + (ok === null ? "idle" : ok ? "ok" : "bad");
    p.querySelector("span").textContent = msg;
  }

  /* ---------------- 开始 / 停止 ---------------- */
  function collectConfig() {
    return {
      csv_path: $("#csv-path").value.trim(),
      output: $("#output").value.trim(),
      years: Array.from(document.querySelectorAll("#years input:checked"))
        .map((i) => parseInt(i.value, 10)),
      priority_year: 2025,
      workers: parseInt($("#workers").value, 10) || 4,
      prefer: $("#prefer").value,
      token: $("#token").value.trim(),
      overwrite: $("#overwrite").checked,
      resume: $("#resume").checked,
      save_fulltext: $("#save-fulltext").checked,
      ai_enabled: $("#ai-enabled").checked,
      deepseek_key: $("#deepseek-key").value.trim(),
      deepseek_model: $("#deepseek-model").value,
      deepseek_advanced_model: $("#deepseek-advanced-model").value,
      ai_summary: $("#ai-summary").checked,
      ai_verify: $("#ai-verify").checked,
    };
  }

  $("#btn-start").addEventListener("click", async () => {
    const cfg = collectConfig();
    saveSettings();
    if (!cfg.years.length) { alert("请至少选择一个报告年度"); return; }
    const { status, body } = await post("/api/start", cfg);
    if (status !== 200) { alert(body.msg || "启动失败"); return; }
    logOffset = 0;
    $("#log").innerHTML = "";
    switchView("log");
    startPolling(900);
  });

  $("#btn-stop").addEventListener("click", async () => {
    $("#btn-stop").disabled = true;
    await post("/api/stop", {});
  });

  $("#btn-retry").addEventListener("click", async () => {
    const r = await post("/api/retry", {});
    if (!r.body.ok) { alert(r.body.msg || "无法重跑"); return; }
    const years = Array.from(new Set(r.body.keys.map((k) => parseInt(k.split("|")[1], 10))));
    const cfg = collectConfig();
    cfg.years = years;
    cfg.retry_keys = r.body.keys;
    const st = await post("/api/start", cfg);
    if (st.status !== 200) { alert(st.body.msg || "启动失败"); return; }
    logOffset = 0; $("#log").innerHTML = ""; switchView("log"); startPolling(900);
  });

  $("#btn-reveal").addEventListener("click", async () => {
    const { body } = await post("/api/reveal", {});
    if (!body.ok) alert(body.msg || "无法打开目录");
  });

  $("#btn-xlsx").addEventListener("click", () => {
    window.location.href = "/api/download?kind=xlsx";
  });

  /* ---------------- 视图切换 ---------------- */
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchView(t.dataset.view)));
  function switchView(v) {
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.view === v));
    ["matrix", "issues", "log"].forEach((k) =>
      $("#view-" + k).classList.toggle("hidden", k !== v));
  }

  /* ---------------- 轮询 ---------------- */
  function startPolling(interval) {
    if (timer) clearInterval(timer);
    tick();
    timer = setInterval(tick, interval);
  }

  async function tick() {
    let body;
    try {
      const r = await api("/api/status?log_from=" + logOffset);
      body = r.body;
    } catch (e) { return; }
    if (!body) return;

    if (body.connection && body.connection.ok !== null && body.connection.ok !== undefined) {
      setConn(body.connection.ok, body.connection.msg);
    }
    if (body.ai_connection && body.ai_connection.ok !== null && body.ai_connection.ok !== undefined) {
      setAiConn(body.ai_connection.ok, body.ai_connection.msg);
    }

    // 日志增量
    if (body.logs && body.logs.length) {
      const box = $("#log");
      const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 40;
      box.insertAdjacentHTML("beforeend", body.logs.map((l) =>
        `<div class="l"><span class="t">${esc(l.t)}</span><span class="${esc(l.level)}">${esc(l.msg)}</span></div>`
      ).join(""));
      logOffset = body.log_total;
      if (atBottom) box.scrollTop = box.scrollHeight;
    }

    // 统计
    const ok = (body.records || []).filter((r) => r.status === "OK").length;
    $("#s-total").textContent = body.total || 0;
    $("#s-done").textContent = body.done || 0;
    $("#s-ok").textContent = ok;
    $("#s-bad").textContent = body.abnormal || 0;
    if ($("#s-rdna")) $("#s-rdna").textContent = body.rd_na || 0;
    if ($("#s-vbad")) $("#s-vbad").textContent = body.verify_bad || 0;
    const pct = body.total ? Math.round((body.done / body.total) * 100) : 0;
    $("#bar").style.width = pct + "%";
    $("#progress-text").textContent = body.running
      ? `处理中 ${body.done}/${body.total}（${pct}%）` +
        (body.resumed ? `，已复用 ${body.resumed}` : "")
      : body.finished_at
        ? `已结束 ${body.finished_at}　共 ${body.total} 个任务` +
          (body.resumed ? `，断点复用 ${body.resumed}` : "")
        : "尚未开始";

    $("#btn-start").disabled = !!body.running;
    $("#btn-stop").disabled = !body.running;
    $("#btn-xlsx").disabled = !(body.exports && body.exports.xlsx);
    $("#btn-retry").disabled = !!body.running || !(body.abnormal > 0);

    if (body.error) $("#progress-text").textContent = "运行失败：" + body.error;

    // 表格
    if (JSON.stringify(body.records) !== JSON.stringify(lastRecords)) {
      lastRecords = body.records || [];
      renderMatrix(lastRecords);
      renderIssues(lastRecords);
      if (current) refreshDrawerMeta();
    }

    if (!body.running && timer) {
      clearInterval(timer);
      timer = setInterval(tick, 5000);
    }
  }

  /* ---------------- 采集矩阵 ---------------- */
  function renderMatrix(records) {
    const table = $("#matrix");
    if (!records.length) return;
    const years = Array.from(new Set(records.map((r) => r.year))).sort((a, b) => b - a);
    const byCo = new Map();
    records.forEach((r) => {
      if (!byCo.has(r.ts_code)) byCo.set(r.ts_code, { name: r.name, years: {} });
      byCo.get(r.ts_code).years[r.year] = r;
    });

    table.querySelector("thead").innerHTML =
      "<tr><th class='c-co'>公司</th>" +
      years.map((y) => `<th>${y} 年年报</th>`).join("") + "</tr>";

    const rows = [];
    byCo.forEach((co, code) => {
      const cells = years.map((y) => {
        const r = co.years[y];
        if (!r) return "<td></td>";
        const t = tone(r.status);
        const slices = KEYS.map((k) => {
          const s = (r.sections || {})[k] || {};
          const cls = !s.found ? ""
            : s.status === "NA" ? "na"
            : s.status === "AI_SUMMARY" ? "sum"
            : s.ai ? "ai" : s.loose ? "loose" : "on";
          const tag = s.status === "NA" ? "（公司填报不适用）"
            : s.status === "AI_SUMMARY" ? "（AI 概括生成，非原文）"
            : s.ai ? "（AI 定位）" : s.loose ? "（疑似，待复核）" : "";
          const vd = ((r.verify || {})[k] || {}).verdict;
          const vtag = vd && vd !== "skip" ? `｜后验：${vd}` : "";
          return `<i class="slice ${k === "mdna" ? "mdna " : ""}${cls}" title="${esc(NAMES[k])}${tag}：${esc(s.how || "未识别")}${esc(vtag)}"></i>`;
        }).join("");
        return `<td><button class="cell" data-key="${esc(r.key)}">
            <span class="cell-head"><i class="dot ${t}"></i>
              <span class="st ${t === "bad" ? "bad" : t === "warn" ? "warn" : ""}">${esc(LABEL[r.status] || r.status)}</span></span>
            <span class="slices">${slices}</span>
          </button></td>`;
      }).join("");
      rows.push(`<tr><td class="c-co"><div class="co-name">${esc(co.name)}</div>
        <div class="co-code">${esc(code)}</div></td>${cells}</tr>`);
    });
    table.querySelector("tbody").innerHTML = rows.join("");

    table.querySelectorAll(".cell").forEach((btn) =>
      btn.addEventListener("click", () => openDrawer(btn.dataset.key)));
  }

  function renderIssues(records) {
    const bad = records.filter((r) => BAD.includes(r.status) || WARN.includes(r.status));
    const tb = $("#issues").querySelector("tbody");
    if (!bad.length) {
      tb.innerHTML = "<tr><td colspan='4' class='empty'>暂无异常记录。</td></tr>";
      return;
    }
    tb.innerHTML = bad.map((r) => `<tr>
      <td><div class="co-name">${esc(r.name)}</div><div class="co-code">${esc(r.ts_code)}</div></td>
      <td class="co-code">${r.year}</td>
      <td><span class="st ${tone(r.status)}">${esc(LABEL[r.status] || r.status)}</span></td>
      <td>${esc(r.message)}</td></tr>`).join("");
  }

  /* ---------------- 抽屉 ---------------- */
  function openDrawer(key) {
    current = lastRecords.find((r) => r.key === key);
    if (!current) return;
    $("#drawer").classList.remove("hidden");
    $("#scrim").classList.remove("hidden");
    refreshDrawerMeta();
    const first = KEYS.find((k) => (current.sections || {})[k] && current.sections[k].found);
    if (first) loadSection(first); else $("#dw-text").textContent = "该年报没有切出任何章节。";
  }
  function closeDrawer() {
    current = null;
    $("#drawer").classList.add("hidden");
    $("#scrim").classList.add("hidden");
  }
  $("#dw-close").addEventListener("click", closeDrawer);
  $("#scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

  function refreshDrawerMeta() {
    const r = lastRecords.find((x) => x.key === (current && current.key));
    if (!r) return;
    current = r;
    $("#dw-eyebrow").textContent = `${r.year} 年年度报告`;
    $("#dw-title").textContent = `${r.name}　${r.ts_code}`;
    const rows = [
      ["状态", LABEL[r.status] || r.status],
      ["断点恢复", r.resumed ? `是（${r.resume_source || "历史成果"}）` : "否"],
      ["公告标题", r.title || "—"],
      ["公告日期", r.ann_date || "—"],
      ["来源", r.source || "—"],
      ["页数 / 引擎", `${r.pages || 0} 页 / ${r.engine || "—"}`],
      ["研发投入", r.rd_status || "—"],
      ["业务概况来源", r.business_origin || "—"],
      ["切片后验", r.verify_detail || "未后验"],
      ["后验问题", (r.verify_problems || []).join("；") || "—"],
      ["切片提示", (r.slice_notes || []).join("；") || "—"],
      ["说明", r.message || "—"],
      ["PDF", r.pdf_path || "—"],
    ];
    $("#dw-meta").innerHTML = rows
      .map(([k, v]) => `<div><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join("");
    $("#dw-tabs").innerHTML = KEYS.map((k) => {
      const s = (r.sections || {})[k] || {};
      const dis = s.found ? "" : "disabled";
      const mark = s.status === "NA" ? "∅"
        : s.status === "AI_SUMMARY" ? "✎"
        : s.ai ? "🤖" : s.loose ? "≈" : s.found ? "" : "×";
      return `<button data-k="${k}" ${dis}>${esc(NAMES[k])} ${mark}
        <span class="co-code">${s.chars || 0}</span></button>`;
    }).join("");
    $("#dw-tabs").querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => loadSection(b.dataset.k)));
  }

  async function loadSection(k) {
    if (!current) return;
    $("#dw-tabs").querySelectorAll("button").forEach((b) =>
      b.classList.toggle("active", b.dataset.k === k));
    $("#dw-text").textContent = "读取中…";
    const q = new URLSearchParams({
      ts_code: current.ts_code, name: current.name,
      year: current.year, key: k,
    });
    const { body } = await api("/api/preview?" + q.toString());
    $("#dw-text").textContent = body.ok
      ? body.text + (body.truncated ? "\n\n…（内容过长，此处仅预览前 60000 字，完整内容见输出目录）" : "")
      : "无法读取：" + (body.msg || "未知原因");
  }

  /* ---------------- 初始化 ---------------- */
  startPolling(4000);
})();
