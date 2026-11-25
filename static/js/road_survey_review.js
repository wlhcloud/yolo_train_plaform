class RoadSurveyReviewUI {
  constructor(app) {
    this.app = app;
    this.$ = app.$;
    this.dialog = this.$("reviewDialog");
    this.current = null;
    this.videoId = null;
    this.selectionVersion = 0;
    this.changing = false;
    const action = (id, fn) =>
      this.$(id).addEventListener("click", () => this.guard(fn));
    action("reviewTask", () => this.open());
    action("closeReview", () => this.dialog.close());
    action("generateReview", () => this.generate());
    action("cancelReview", async () => {
      if (this.current) {
        this.current = (
          await this.app.api(
            `/${this.taskId}/reviews/${this.current.id}/stop`,
            "POST",
          )
        ).review;
        this.render();
      }
    });
    this.$("reviewStream").onchange = () => this.setStart();
    this.$("reviewHistory").onchange = () =>
      this.guard(() => this.choose(this.$("reviewHistory").value));
    this.$("reviewSpeed").onchange = () => {
      this.$("reviewVideo").playbackRate = Number(this.$("reviewSpeed").value);
    };
    this.dialog.addEventListener("close", () => this.$("reviewVideo").pause());
  }
  async guard(fn) {
    try {
      await fn();
    } catch (e) {
      this.$("reviewMessage").textContent = e.message || String(e);
    }
  }
  dateTime(seconds) {
    const d = new Date(this.config.survey_date + "T00:00:00");
    d.setSeconds(seconds);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }
  async open() {
    this.selectionVersion++;
    const task = (await this.app.api(`/${this.app.task.id}`)).task;
    if (!task.manifest) throw Error("请先扫描视频目录");
    this.taskId = task.id;
    this.config = task.config;
    this.manifest = task.manifest;
    this.current = null;
    this.$("reviewStream").innerHTML = this.manifest.streams
      .map(
        (s) =>
          `<option value="${s.key}">${this.app.escape(s.direction)} / ${this.app.escape(s.name)}</option>`,
      )
      .join("");
    this.setStart();
    this.dialog.showModal();
    await this.refreshHistory();
  }
  setStart() {
    const stream = this.manifest.streams.find(
      (s) => s.key === this.$("reviewStream").value,
    );
    if (stream)
      this.$("reviewStart").value = this.dateTime(Math.ceil(Math.max(...this.manifest.streams.map(s => s.files[0].start)) / 300) * 300);
    this.$("reviewMessage").textContent = "";
  }
  async refreshHistory() {
    const version = this.selectionVersion;
    const rows = (await this.app.api(`/${this.taskId}/reviews`)).reviews;
    if (version !== this.selectionVersion) return;
    this.$("reviewHistory").innerHTML =
      rows
        .map(
          (r) =>
            `<option value="${r.id}">${r.whole_road ? "全方向" : this.app.escape(r.direction) + " / " + this.app.escape(r.stream_name)} · ${this.app.formatTime(r.start)} · ${this.app.states[r.status] || r.status}</option>`,
        )
        .join("") || '<option value="">暂无抽查记录</option>';
    const selected =
      rows.find((r) => r.id === this.current?.id) || rows[0] || null;
    this.current = selected;
    if (selected) this.$("reviewHistory").value = selected.id;
    this.render();
  }
  async choose(id) {
    const version = ++this.selectionVersion;
    const review = id
      ? (await this.app.api(`/${this.taskId}/reviews/${id}`)).review
      : null;
    if (version !== this.selectionVersion) return;
    this.current = review;
    this.render();
  }
  async generate() {
    const value = this.$("reviewStart").value;
    const start =
      (new Date(value) - new Date(this.config.survey_date + "T00:00:00")) /
      1000;
    if (!value || !Number.isFinite(start))
      throw Error("请选择有效的抽查起始时间");
    this.selectionVersion++;
    this.changing = true;
    this.$("generateReview").disabled = true;
    try {
      this.current = (
        await this.app.api(`/${this.taskId}/reviews`, "POST", {
          stream: this.$("reviewStream").value,
          start_second: start,
          whole_road: true,
        })
      ).review;
      this.$("reviewMessage").textContent =
        "已加入队列，生成期间可关闭窗口，稍后重新打开查看。";
      await this.refreshHistory();
    } finally {
      this.changing = false;
      this.render();
    }
  }
  async poll() {
    if (
      this.changing ||
      !this.current ||
      !["queued", "running", "stopping"].includes(this.current.status)
    )
      return;
    const version = this.selectionVersion;
    const id = this.current.id;
    const before = this.current.status;
    const review = (await this.app.api(`/${this.taskId}/reviews/${id}`)).review;
    if (version !== this.selectionVersion || this.current?.id !== id) return;
    this.current = review;
    this.render();
    if (before !== this.current.status) await this.refreshHistory();
  }
  render() {
    const review = this.current,
      busy =
        review && ["queued", "running", "stopping"].includes(review.status);
    this.$("cancelReview").disabled = !busy;
    this.$("generateReview").disabled = !!busy;
    this.$("reviewProgress").textContent = review
      ? `${this.app.states[review.status] || review.status}${review.whole_road ? " · 全方向" : " · 单机位旧记录"}${review.progress.part ? " · 机位 " + review.progress.part + "/" + review.progress.parts : ""}${review.progress.seconds !== undefined ? " · 已处理 " + review.progress.seconds.toFixed(1) + " / " + review.progress.duration.toFixed(1) + " 秒" : ""}${review.error ? " · " + review.error : ""}`
      : "选择时段后点击生成";
    const ready = review?.status === "finished";
    if (ready && this.$("reviewMessage").textContent.startsWith("已加入队列")) {
      this.$("reviewMessage").textContent =
        "抽查回放已生成，可按原速播放、暂停、拖动或慢放。";
    }
    this.$("reviewPlayerPanel").hidden = !ready;
    if (!ready) {
      this.$("reviewVideo").pause();
      this.$("reviewVideo").removeAttribute("src");
      this.$("reviewVideo").load();
      this.videoId = null;
      return;
    }
    const base = this.app.base + `/${this.taskId}/reviews/${review.id}`;
    if (this.videoId !== review.id) {
      this.$("reviewVideo").src = base + "/video";
      this.$("reviewVideo").load();
      this.$("reviewSpeed").value = "1";
      this.$("reviewVideo").playbackRate = 1;
      this.videoId = review.id;
    }
    this.$("reviewVideoDownload").href = base + "/video?download=1";
    this.$("reviewAuditDownload").href = base + "/audit";
    this.$("reviewExcelDownload").href = base + "/download";
    this.$("reviewExcelDownload").textContent = review.whole_road ? "导出全方向5分钟样板ZIP" : "导出单机位旧样板ZIP";
    const picker = this.$("reviewPlaybackPart");
    if (picker.dataset.review !== review.id) {
      picker.innerHTML = (review.parts || []).map(p => `<option value="${this.app.escape(p.key)}">${this.app.escape(p.direction)} / ${this.app.escape(p.name)}</option>`).join("");
      picker.dataset.review = review.id;
      picker.onchange = () => {
        this.$("reviewVideo").src = base + "/video?part=" + encodeURIComponent(picker.value);
        this.$("reviewVideoDownload").href = base + "/video?download=1&part=" + encodeURIComponent(picker.value);
        this.$("reviewVideo").load();
      };
    }
    picker.hidden = !(review.parts || []).length;
    if (picker.value) this.$("reviewVideoDownload").href = base + "/video?download=1&part=" + encodeURIComponent(picker.value);
    const counts = review.result.counts || {};
    const labels = [
      ...new Set([
        ...this.app.labels,
        ...Object.keys(counts["主路"] || {}),
        ...Object.keys(counts["辅路"] || {}),
      ]),
    ];
    this.$("reviewExcelPreview").innerHTML = labels
      .map((name) => {
        const main = counts["主路"]?.[name] || 0,
          side = counts["辅路"]?.[name] || 0;
        return `<tr><td>${this.app.escape(name)}</td><td>${this.app.escape(review.mapping?.[name] || "")}</td><td>${main}</td><td>${side}</td><td>${main + side}</td></tr>`;
      })
      .join("");
    this.$("reviewCoverage").textContent =
      `回放时长 ${review.result.duration.toFixed(1)} 秒，录像覆盖 ${review.result.coverage_seconds.toFixed(1)} 秒。视频按真实时间播放，停录间隔保留；本片段为独立抽查复算。`;
  }
}
