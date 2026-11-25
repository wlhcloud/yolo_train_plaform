/* Road-survey editor. Coordinates are normalized and remain tied to one camera. */
class RoadSurveyApp {
  constructor(options) {
    Object.assign(this, options);
    this.base = `/api/project/${this.projectId}/road-surveys`;
    this.task = null;
    this.busyStates = [
      "queued",
      "running",
      "scan_queued",
      "scanning",
      "stopping",
    ];
    this.states = {
      draft: "草稿",
      ready: "待运行",
      queued: "排队中",
      running: "推理中",
      scan_queued: "扫描排队",
      scanning: "扫描中",
      stopping: "正在停止",
      stopped: "已停止",
      interrupted: "运行中断",
      failed: "失败",
      finished: "已完成",
    };
    this.$ = (id) => document.getElementById(id);
    const action = (id, fn) =>
      this.$(id).addEventListener("click", () => this.guard(fn));
    action("newTask", () => this.newTask());
    action("discover", () => this.discover());
    action("saveTask", () => this.save());
    action("scanTask", async () => {
      await this.save();
      this.task = (await this.api(`/${this.task.id}/scan`, "POST")).task;
      this.renderState();
    });
    action("startTask", () => this.start(true));
    action("rerunTask", () => this.start(false));
    action("stopTask", async () => {
      if (this.task) {
        this.task = (await this.api(`/${this.task.id}/stop`, "POST")).task;
        this.renderState();
      }
    });
    action("liveTask", () => this.openLive());
    this.$("closeLive").onclick = () => this.$("liveDialog").close();
    this.$("liveDialog").addEventListener("close", () => {
      this.$("liveImage").removeAttribute("src");
      this.liveStreamKey = null;
    });
    action("addDirection", () => {
      this.readForm();
      if (this.config.directions.length >= 2) throw Error("最多两个方向");
      this.config.directions.push({
        name: "南向北",
        streams: [this.blankStream()],
      });
      this.renderForm();
    });
    this.$("directions").addEventListener("click", (e) =>
      this.guard(() => this.directionAction(e)),
    );
    this.$("timeline").addEventListener("click", (e) => {
      if (e.target.dataset.editRegion)
        this.guard(() => this.openRegion(e.target.dataset.editRegion));
    });
    this.$("taskList").addEventListener("click", (e) => {
      const b = e.target.closest("[data-task]");
      if (b) this.guard(() => this.load(Number(b.dataset.task)));
    });
    this.$("discovered").addEventListener("click", (e) => {
      const b = e.target.closest("[data-road]");
      if (b) {
        this.guard(() => this.selectRoad(this.roads[Number(b.dataset.road)]));
      }
    });
    this.$("closeRegion").onclick = () => this.$("regionDialog").close();
    action("loadFrame", () => this.loadFrame());
    this.$("regionRole").onchange = () => {
      this.mode = null;
      this.updateCountMode();
      this.drawCanvas();
    };
    this.$("regionCountMode").onchange = () => {
      this.region().mode = this.$("regionCountMode").value;
      this.updateCountMode();
      this.drawCanvas();
    };
    action("importMaterialRegion", () => {
      const item =
        this.materialRegions[Number(this.$("materialRegionSource").value)];
      if (!item) throw Error("请先在素材配置页画好区域，再选择要导入的区域");
      this.editRegions[this.$("regionRole").value] = {
        polygon: structuredClone(item.polygon),
        mode: "region",
        source_material_id: item.material_id,
        source_annotation_index: item.annotation_index,
      };
      this.mode = null;
      this.updateCountMode();
      this.drawCanvas();
      this.$("drawingHint").textContent =
        `已导入素材 ${item.material_id} 的“${item.label}”。请对照当前视频画面确认是同一机位，再保存。`;
    });
    for (const [id, mode] of [
      ["drawPolygon", "polygon"],
      ["drawLine", "line"],
      ["drawArrow", "arrow"],
    ])
      action(id, () => {
        this.mode = mode;
        this.region()[mode] = [];
        this.drawCanvas();
        this.$("drawingHint").textContent =
          mode === "polygon"
            ? "沿边界依次点击，最后点击“完成多边形”。"
            : "在画面上点击两个点。通行箭头从第一个点指向第二个点。";
      });
    action("finishPolygon", () => {
      if ((this.region().polygon || []).length < 3) throw Error("至少三个顶点");
      this.mode = null;
      this.drawCanvas();
    });
    action("removeRegion", () => {
      delete this.editRegions[this.$("regionRole").value];
      this.mode = null;
      this.drawCanvas();
    });
    action("saveRegions", async () => {
      const [di, si] = this.editKey.split(":").map(Number);
      this.config.directions[di].streams[si].regions = structuredClone(
        this.editRegions,
      );
      await this.save();
      this.$("regionDialog").close();
    });
    this.$("regionCanvas").addEventListener("click", (e) => {
      if (!this.mode || !this.frameImage) return;
      const r = e.target.getBoundingClientRect();
      const points =
        this.region()[this.mode] || (this.region()[this.mode] = []);
      points.push([
        Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
        Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
      ]);
      if (this.mode !== "polygon" && points.length === 2) this.mode = null;
      this.drawCanvas();
    });
    this.newTask(undefined, false);
    this.guard(() => this.restoreSelection());
    this.guard(() => this.refreshRuntime());
    this.timer = setInterval(() => this.guard(() => this.poll()), 2000);
  }
  escape(value) {
    return String(value ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  }
  message(value, kind = "danger") {
    this.$("notice").textContent = value;
    this.$("notice").className = `survey-message alert alert-${kind}`;
  }
  async guard(fn) {
    try {
      return await fn();
    } catch (e) {
      this.message(e.message || String(e));
    }
  }
  async api(path = "", method = "GET", body) {
    const r = await fetch(this.base + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    let value;
    try {
      value = await r.json();
    } catch {
      throw Error(`请求失败 (${r.status})，请确认本地服务正常`);
    }
    if (!r.ok) throw Error(value.message || `请求失败 (${r.status})`);
    return value;
  }
  blankStream() {
    return {
      name: "机位1",
      start_time: "07:00:00",
      cards: [{ folder: "", gap_seconds: 0 }],
      regions: {},
    };
  }
  newTask(road, clearSelection = true) {
    if (clearSelection) this.rememberSelection(null);
    this.$("saveState").textContent = "未保存";
    this.task = null;
    this.manifest = null;
    this.config = {
      road_name: road?.road_name || "",
      road_root: road?.root || "",
      spot_code: "",
      survey_date: new Date().toLocaleDateString("sv-SE"),
      system_model: "best.pt",
      inference_resolution: "640",
      inference_batch_size: 0,
      class_mapping: Object.fromEntries(
        this.labels.map((n) => [n, this.categories.includes(n) ? n : ""]),
      ),
      directions: road
        ? structuredClone(road.directions)
        : [
            { name: "北向南", streams: [this.blankStream()] },
            { name: "南向北", streams: [this.blankStream()] },
          ],
    };
    this.$("timelineConfirmed").checked = false;
    this.renderForm();
    this.renderTimeline();
    this.renderState();
  }
  readForm() {
    this.config.road_name = this.$("roadName").value.trim();
    this.config.spot_code = this.$("spotCode").value.trim();
    this.config.survey_date = this.$("surveyDate").value;
    this.config.system_model = this.$("surveyModel").value;
    this.config.inference_resolution = this.$("inferenceResolution").value;
    this.config.inference_batch_size = Number(
      this.$("inferenceBatchSize").value,
    );
    document.querySelectorAll("[data-config-path]").forEach((el) => {
      const parts = el.dataset.configPath.split(".");
      let target = this.config;
      for (const key of parts.slice(0, -1)) target = target[key];
      target[parts.at(-1)] = el.type === "number" ? Number(el.value) : el.value;
    });
    document.querySelectorAll("[data-class]").forEach((el) => {
      this.config.class_mapping[el.dataset.class] = el.value;
    });
    return this.config;
  }
  input(path, value, type = "text", label = "") {
    return `<label class="flex-grow-1">${label}<input class="form-control form-control-sm" data-config-path="${path}" type="${type}" value="${this.escape(value)}" ${type === "number" ? 'min="0" step="0.1"' : ""}></label>`;
  }
  renderForm() {
    this.$("roadName").value = this.config.road_name;
    this.$("spotCode").value = this.config.spot_code;
    this.$("surveyDate").value = this.config.survey_date;
    this.$("surveyModel").value = this.config.system_model || "best.pt";
    this.$("inferenceResolution").value = String(
      this.config.inference_resolution || "640",
    );
    this.$("inferenceBatchSize").value = this.config.inference_batch_size || 0;
    this.$("directions").innerHTML = this.config.directions
      .map(
        (d, di) =>
          `<section class="mb-4"><div class="survey-toolbar">${this.input(`directions.${di}.name`, d.name, "text", "方向名称")}<button class="btn btn-outline-secondary btn-sm" data-action="add-stream" data-di="${di}">添加机位序列</button><button class="btn btn-outline-danger btn-sm" data-action="remove-direction" data-di="${di}">移除方向</button></div>${d.streams.map((s, si) => `<div class="stream-panel"><div class="survey-toolbar">${this.input(`directions.${di}.streams.${si}.name`, s.name, "text", "机位序列名称")}${this.input(`directions.${di}.streams.${si}.start_time`, s.start_time, "time", "首段拍摄起始时间")}<button class="btn btn-outline-danger btn-sm" data-action="remove-stream" data-di="${di}" data-si="${si}">移除机位</button></div><div class="text-hint mt-2">视频目录按下列顺序处理；可选普通视频目录或一卡、二卡目录。间隔为下一目录开始前的停录秒数，连续录像填 0。</div>${s.cards.map((c, ci) => `<div class="cards-row">${this.input(`directions.${di}.streams.${si}.cards.${ci}.folder`, c.folder, "text", `第${ci + 1}个视频目录`)}${this.input(`directions.${di}.streams.${si}.cards.${ci}.gap_seconds`, c.gap_seconds, "number", "开始前间隔（秒）")}<div class="survey-toolbar align-self-end">${["up", "down", "remove-card"].map((a, i) => `<button class="btn btn-outline-secondary btn-sm" data-action="${a}" data-di="${di}" data-si="${si}" data-ci="${ci}">${["上移", "下移", "移除"][i]}</button>`).join("")}</div></div>`).join("")}<button class="btn btn-outline-primary btn-sm mt-2" data-action="add-card" data-di="${di}" data-si="${si}">添加下一视频目录</button><div class="text-hint mt-2">已配置区域：${this.escape(Object.keys(s.regions || {}).join("、") || "未配置，请扫描后画区域")}</div></div>`).join("")}</section>`,
      )
      .join("");
    this.$("classMapping").innerHTML = this.labels
      .map(
        (name) =>
          `<label>${this.escape(name)}<select class="form-select form-select-sm" data-class="${this.escape(name)}"><option value="">请选择归类</option>${[...this.categories, this.exclude].map((c) => `<option value="${c}" ${this.config.class_mapping[name] === c ? "selected" : ""}>${c}</option>`).join("")}</select></label>`,
      )
      .join("");
    this.renderState();
  }
  directionAction(e) {
    const b = e.target.closest("[data-action]");
    if (!b) return;
    this.readForm();
    const di = Number(b.dataset.di),
      si = Number(b.dataset.si),
      ci = Number(b.dataset.ci),
      d = this.config.directions[di],
      s = d.streams[si];
    switch (b.dataset.action) {
      case "add-stream":
        d.streams.push(this.blankStream());
        break;
      case "remove-direction":
        this.config.directions.splice(di, 1);
        break;
      case "remove-stream":
        d.streams.splice(si, 1);
        break;
      case "add-card":
        s.cards.push({ folder: "", gap_seconds: 0 });
        break;
      case "remove-card":
        s.cards.splice(ci, 1);
        break;
      case "up":
        if (ci > 0)
          [s.cards[ci - 1], s.cards[ci]] = [s.cards[ci], s.cards[ci - 1]];
        break;
      case "down":
        if (ci < s.cards.length - 1)
          [s.cards[ci + 1], s.cards[ci]] = [s.cards[ci], s.cards[ci + 1]];
        break;
    }
    this.$("timelineConfirmed").checked = false;
    this.renderForm();
  }
  async discover() {
    this.roads = (
      await this.api("/discover", "POST", {
        root: this.$("discoverRoot").value,
      })
    ).roads;
    this.$("discovered").innerHTML =
      this.roads
        .map(
          (r, i) =>
            `<button class="btn btn-sm btn-outline-secondary text-start w-100 mb-2" data-road="${i}">${this.escape(r.road_name)}</button>`,
        )
        .join("") || "未发现包含方向名称的道路目录";
  }
  async selectRoad(road) {
    const query = new URLSearchParams({
      road_root: road.root,
      road_name: road.road_name,
    });
    const { tasks } = await this.api(`/find?${query}`);
    if (tasks.length === 1) {
      await this.load(tasks[0].id);
      this.message(`已加载“${road.road_name}”保存的配置。`, "info");
    } else if (tasks.length > 1) {
      this.newTask(road);
      this.$("taskList").innerHTML = tasks
        .map(
          (t) =>
            `<button class="task-item" data-task="${t.id}">${this.escape(t.name)} · #${t.id}<br><span class="status-pill">${this.states[t.status] || t.status}</span></button>`,
        )
        .join("");
      this.message(
        "这条道路有多份已保存的调查任务，请从左侧选择要继续的任务。",
        "info",
      );
    } else {
      this.newTask(road);
      this.message(
        "这条道路还没有保存的配置，已新建草稿，请核实日期、起始时间和卡顺序。",
        "info",
      );
    }
  }
  async refreshList() {
    const { tasks } = await this.api();
    this.$("taskList").innerHTML =
      tasks
        .map(
          (t) =>
            `<button class="task-item ${t.id === this.task?.id ? "active" : ""}" data-task="${t.id}">${this.escape(t.name)}<br><span class="status-pill">${this.states[t.status] || t.status}</span></button>`,
        )
        .join("") || '<p class="text-hint">还没有道路任务</p>';
    return tasks;
  }
  rememberSelection(id) {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set("task", String(id));
    else url.searchParams.delete("task");
    history.replaceState(null, "", url);
    try {
      const key = `road-survey-selected-${this.projectId}`;
      if (id) localStorage.setItem(key, String(id));
      else localStorage.removeItem(key);
    } catch {
      /* A task URL also works when local storage is disabled. */
    }
  }
  async restoreSelection() {
    const tasks = await this.refreshList();
    let id = new URL(window.location.href).searchParams.get("task");
    if (!id) {
      try {
        id = localStorage.getItem(`road-survey-selected-${this.projectId}`);
      } catch {
        /* optional */
      }
    }
    const selected = tasks.find((t) => t.id === Number(id)) || tasks[0];
    if (selected) await this.load(selected.id);
  }
  async load(id) {
    this.task = (await this.api(`/${id}`)).task;
    this.config = structuredClone(this.task.config);
    this.manifest = this.task.manifest;
    this.rememberSelection(this.task.id);
    this.$("saveState").textContent = `已保存 · 任务 #${this.task.id}`;
    this.$("timelineConfirmed").checked = false;
    this.renderForm();
    this.renderTimeline();
    this.renderState();
    await this.refreshList();
  }
  async save() {
    this.readForm();
    const creating = !this.task;
    const r = await this.api(
      creating ? "" : `/${this.task.id}`,
      creating ? "POST" : "PUT",
      this.config,
    );
    this.task = r.task;
    this.rememberSelection(this.task.id);
    this.manifest = r.task.manifest;
    this.$("saveState").textContent = `已保存 · 任务 #${this.task.id}`;
    this.renderTimeline();
    this.renderState();
    await this.refreshList();
    return r;
  }
  formatTime(seconds) {
    const day = Math.floor(seconds / 86400),
      n = Math.floor(seconds % 86400);
    return `${day ? "次日+" + (day - 1) + " " : ""}${String(Math.floor(n / 3600)).padStart(2, "0")}:${String(Math.floor(n / 60) % 60).padStart(2, "0")}:${String(n % 60).padStart(2, "0")}`;
  }
  sourceLabel(stream, file) {
    const source = this.config.directions[stream.direction_index]?.streams[stream.stream_index];
    const folder = source?.cards?.[file.card - 1]?.folder;
    if (!folder) return file.card_label || "视频目录";
    const name = folder.replace(/[\\/]+$/, "").split(/[\\/]/).at(-1);
    const match = name.match(/^(?:第)?([1-9]|[1-5][0-9]|6[0-4]|[一二三四五六七八九十])(?:张)?卡?$/) || name.match(/^卡([1-9]|[1-5][0-9]|6[0-4]|[一二三四五六七八九十])$/);
    if (!match) return "视频目录";
    const n = Number(match[1]);
    return (n >= 1 && n <= 10 ? "一二三四五六七八九十"[n - 1] : match[1]) + "卡";
  }
  renderTimeline() {
    if (!this.manifest) {
      this.$("timeline").innerHTML =
        '<p class="text-hint">保存配置并扫描后，可预览时间表、抽帧配置主辅路区域。</p>';
      return;
    }
    this.$("timeline").innerHTML =
      `<p>共 ${this.manifest.file_count} 段视频，累计录像 ${(this.manifest.duration / 3600).toFixed(2)} 小时。</p>` +
      this.manifest.streams
        .map(
          (s) =>
            `<details class="mb-3" open><summary>${this.escape(s.direction)} / ${this.escape(s.name)}（${s.files.length}段）</summary><button class="btn btn-outline-primary btn-sm my-2" data-edit-region="${s.key}">核对画面 / 配置主辅路区域</button><div class="survey-timeline"><table class="table table-sm"><thead><tr><th>顺序</th><th>来源</th><th>文件</th><th>时长</th><th>起始时间</th><th>结束时间</th></tr></thead><tbody>${s.files.map((f, i) => `<tr><td>${i + 1}</td><td>${this.escape(this.sourceLabel(s, f))}</td><td title="${this.escape(f.path)}">${this.escape(f.name)}</td><td>${f.duration.toFixed(2)}秒</td><td>${this.formatTime(f.start)}</td><td>${this.formatTime(f.end)}</td></tr>`).join("")}</tbody></table></div></details>`,
        )
        .join("");
  }
  async start(resume) {
    if (!this.$("timelineConfirmed").checked)
      throw Error("请先核对并勾选视频时间表确认");
    this.readForm();
    const missing = this.labels.filter((n) => !this.config.class_mapping[n]);
    if (missing.length) throw Error("请设置车型归类：" + missing.join("、"));
    await this.save();
    if (!this.manifest) throw Error("视频配置有变化，请重新扫描时间表");
    this.task = (
      await this.api(`/${this.task.id}/start`, "POST", {
        timeline_confirmed: true,
        resume,
      })
    ).task;
    this.renderState();
  }
  renderState() {
    const t = this.task,
      busy = t && this.busyStates.includes(t.status);
    for (const id of [
      "saveTask",
      "scanTask",
      "startTask",
      "rerunTask",
      "addDirection",
    ])
      this.$(id).disabled = !!busy;
    this.$("stopTask").disabled = !busy;
    this.$("liveTask").disabled = !t;
    this.$("reviewTask").disabled = !t || !this.manifest;
    if (this.$("liveDialog").open) this.renderLive();
    this.$("progress").textContent = t
      ? `${this.states[t.status] || t.status}${t.progress?.phase === "resuming" ? " · 正在解码定位帧断点（不重复推理）" : ""}${t.progress?.files ? " · " + t.progress.files + "个视频" : ""}${t.progress?.current_file ? " · " + t.progress.current_file.split(/[\\/]/).at(-1) : ""}${t.progress?.frame ? " · " + t.progress.frame + "/" + t.progress.estimated_frames + "帧" : ""}${t.progress?.counts !== undefined ? " · 已计数 " + t.progress.counts : ""}${t.error ? "\n" + t.error : ""}`
      : "尚未保存任务";
    this.$("progress").className =
      "survey-message alert " + (t?.error ? "alert-danger" : "alert-light");
    this.$("exports").innerHTML =
      t?.status === "finished"
        ? `<a class="btn btn-primary mb-2" href="${this.base}/${t.id}/download">下载全部表格与计数明细</a><p class="text-hint">黄色行表示录像不足5分钟，单元格批注记录覆盖秒数；完全未覆盖的时段留空，不作为零流量。</p>${(t.result?.exports || []).map((e, i) => `<div class="export-box"><a href="${this.base}/${t.id}/download?index=${i}">${this.escape(e.name)}</a><div class="text-hint">已观测计数 ${e.total} · 不完整时段 ${e.partial_slots}</div></div>`).join("")}<div class="text-hint">明确排除的车型计数事件：${t.result?.excluded || 0}（明细仍保留）。</div>`
        : "";
  }
  openLive() {
    if (!this.task) throw Error("请先选择已保存的道路调查任务");
    this.$("liveDialog").showModal();
    this.renderLive();
  }
  renderLive() {
    const task = this.task,
      progress = task.progress || {};
    const key = `${task.id}:${task.status}`;
    if (this.liveStreamKey !== key) {
      this.$("liveImage").src = `${this.base}/${task.id}/live?t=${Date.now()}`;
      this.liveStreamKey = key;
    }
    this.$("liveStatus").textContent =
      `${this.states[task.status] || task.status} · ${progress.direction || ""} ${progress.stream_name || ""}${progress.current_file ? " · " + progress.current_file.split(/[\\/]/).at(-1) : ""}${task.status === "running" ? "" : " · 开始调查后显示当前处理画面"}`;
    const counts = progress.label_counts || {};
    const labels = [
      ...new Set([
        ...this.labels,
        ...Object.keys(counts["主路"] || {}),
        ...Object.keys(counts["辅路"] || {}),
      ]),
    ];
    this.$("liveCounts").innerHTML = labels
      .map((name) => {
        const main = counts["主路"]?.[name] || 0,
          side = counts["辅路"]?.[name] || 0;
        return `<tr><td>${this.escape(name)}</td><td>${main}</td><td>${side}</td><td>${main + side}</td></tr>`;
      })
      .join("");
    const main = Object.values(counts["主路"] || {}).reduce((a, b) => a + b, 0),
      side = Object.values(counts["辅路"] || {}).reduce((a, b) => a + b, 0);
    this.$("liveTotals").innerHTML =
      `<tr><th>合计</th><th>${main}</th><th>${side}</th><th>${main + side}</th></tr>`;
  }
  async poll() {
    await this.refreshRuntime();
    if (!this.task) return;
    const before = this.task.status;
    const t = (await this.api(`/${this.task.id}`)).task;
    this.task = t;
    if (
      this.busyStates.includes(before) &&
      !this.busyStates.includes(t.status)
    ) {
      this.manifest = t.manifest;
      this.renderTimeline();
      await this.refreshList();
    }
    this.renderState();
    if (this.reviewPlayer?.dialog.open) await this.reviewPlayer.poll();
  }
  async refreshRuntime() {
    const { scheduler } = await this.api("/runtime");
    this.$("schedulerStatus").textContent =
      `并行计算名额 ${scheduler.workers} · 运行中 ${scheduler.running} · 排队 ${scheduler.queued} · 扫描中 ${scheduler.scanning}。抽查共享计算名额，目录扫描使用独立队列。`;
  }
  async openRegion(key) {
    await this.save();
    if (!this.manifest) throw Error("视频配置已变化，请重新扫描后配置区域");
    const [di, si] = key.split(":").map(Number);
    this.editKey = key;
    this.editRegions = structuredClone(
      this.config.directions[di].streams[si].regions || {},
    );
    this.mode = null;
    const s = this.manifest.streams.find((s) => s.key === key);
    if (!s) throw Error("请重新扫描");
    this.$("regionTitle").textContent =
      `${s.direction} / ${s.name} · 主辅路配置`;
    this.$("frameFile").innerHTML = s.files
      .map(
        (f, i) =>
          `<option value="${i}">${this.escape(this.sourceLabel(s, f))} / ${this.escape(f.name)}</option>`,
      )
      .join("");
    this.$("regionDialog").showModal();
    this.materialRegions = (await this.api("/material-regions")).regions;
    this.$("materialRegionSource").innerHTML =
      this.materialRegions
        .map(
          (r, i) =>
            `<option value="${i}">${this.escape(r.filename)} · ${this.escape(r.label)} (#${r.material_id})</option>`,
        )
        .join("") || '<option value="">暂无已配置素材区域</option>';
    this.updateCountMode();
    await this.loadFrame();
  }
  updateCountMode() {
    const mode = this.editRegions[this.$("regionRole").value]?.mode || "region";
    this.$("regionCountMode").value = mode;
    this.$("optionalLineTools").hidden = mode !== "line";
  }
  async loadFrame() {
    const image = new Image();
    image.src = `${this.base}/${this.task.id}/frame?stream=${encodeURIComponent(this.editKey)}&file=${this.$("frameFile").value}&second=0&t=${Date.now()}`;
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(Error("画面加载失败，请确认视频可读"));
    });
    this.frameImage = image;
    this.$("regionCanvas").width = image.naturalWidth;
    this.$("regionCanvas").height = image.naturalHeight;
    this.drawCanvas();
  }
  region() {
    const role = this.$("regionRole").value;
    return (
      this.editRegions[role] ||
      (this.editRegions[role] = {
        polygon: [],
        line: [],
        arrow: [],
        mode: "region",
      })
    );
  }
  drawCanvas() {
    const canvas = this.$("regionCanvas"),
      ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (this.frameImage) ctx.drawImage(this.frameImage, 0, 0);
    for (const [role, r] of Object.entries(this.editRegions)) {
      const color = role === "主路" ? "#34d399" : "#fbbf24";
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3;
      const point = (p) => [p[0] * canvas.width, p[1] * canvas.height];
      if (r.polygon?.length) {
        ctx.beginPath();
        r.polygon.forEach((p, i) => {
          const [x, y] = point(p);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        });
        if (r.polygon.length >= 3) ctx.closePath();
        ctx.stroke();
        ctx.globalAlpha = 0.13;
        ctx.fill();
        ctx.globalAlpha = 1;
        const [x, y] = point(r.polygon[0]);
        ctx.font = "bold 18px sans-serif";
        ctx.fillText(role, x + 5, y + 22);
      }
      for (const key of ["line", "arrow"]) {
        if (r.mode !== "line") continue;
        const points = r[key] || [];
        if (points.length !== 2) continue;
        const [a, b] = points.map(point);
        ctx.strokeStyle = key === "line" ? color : "#fff";
        ctx.beginPath();
        ctx.moveTo(...a);
        ctx.lineTo(...b);
        ctx.stroke();
        if (key === "arrow") {
          const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
          ctx.beginPath();
          ctx.moveTo(...b);
          ctx.lineTo(
            b[0] - 16 * Math.cos(angle - 0.45),
            b[1] - 16 * Math.sin(angle - 0.45),
          );
          ctx.moveTo(...b);
          ctx.lineTo(
            b[0] - 16 * Math.cos(angle + 0.45),
            b[1] - 16 * Math.sin(angle + 0.45),
          );
          ctx.stroke();
        }
      }
    }
  }
}
