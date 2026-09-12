const datasets = {
  zh: window.PROJECT_DATA,
  en: window.PROJECT_DATA_EN,
};

if (!datasets.zh || !datasets.en || !window.UI_COPY) {
  throw new Error("Localized project content is incomplete. Check content.js and i18n.js.");
}

// Each visit starts in English. A language switch applies only to this page view.
let currentLanguage = "en";

let data = datasets[currentLanguage];
let copy = window.UI_COPY[currentLanguage];

function setText(selector, value) {
  document.querySelectorAll(selector).forEach((element) => {
    element.textContent = value;
  });
}

function bindStaticCopy() {
  document.documentElement.lang = currentLanguage === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    if (copy[key]) element.textContent = copy[key];
  });
  document.querySelector(".language-switcher").setAttribute("aria-label", copy.languageLabel);
  document.querySelector(".topbar nav").setAttribute("aria-label", copy.pageNavLabel);
  document.querySelectorAll('.project-logo[href="#top"]').forEach((element) => {
    element.setAttribute("aria-label", copy.backTopLabel);
  });
  document.querySelector('meta[name="description"]').setAttribute("content", copy.pageDescription);
  document.querySelector(".metric-legend").setAttribute("aria-label", currentLanguage === "zh" ? "指标方向说明" : "Metric direction legend");
  document.querySelectorAll("[data-language]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.language === currentLanguage));
  });
}

function bindProjectContent() {
  const project = data.project;
  document.title = `${project.shortName} — Project Page`;
  setText("[data-project-short]", project.shortName);
  setText("[data-kicker]", project.kicker);
  setText("[data-title-en]", project.titleEn);
  setText("[data-title-cn]", project.titleCn);
  setText("[data-affiliation]", project.affiliation);
  setText("[data-demo-intro]", project.demoIntro);
  setText("[data-method-title]", data.method.title);
  setText("[data-method-intro]", data.method.intro);

  document.querySelector("#author-list").innerHTML = project.authors
    .map((author) => `<span>${author.name}${author.note ? `<sup>${author.note}</sup>` : ""}</span>`)
    .join("");
  document.querySelector("#author-notes").textContent = project.authorNote;

  Object.entries(project.links).forEach(([key, url]) => {
    document.querySelectorAll(`[data-link="${key}"]`).forEach((link) => {
      if (url) {
        link.href = url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.removeAttribute("aria-disabled");
      } else {
        link.href = "#";
        link.setAttribute("aria-disabled", "true");
        link.addEventListener("click", (event) => event.preventDefault());
      }
    });
  });
}

function renderHeroEvidence() {
  const project = data.project;
  document.querySelector("#open-line").textContent = project.openLine;
  document.querySelector("#hero-scope").textContent = project.scope;
  document.querySelector("#hero-lead").textContent = project.heroLead;
  document.querySelector("#hero-proof-grid").innerHTML = project.heroHighlights.map((item) => `
    <article class="hero-proof-card">
      <span class="mono">${item.eyebrow}</span>
      <strong>${item.value}</strong>
      <h2>${item.title}</h2>
      <p>${item.text}</p>
      <small>${item.meta}</small>
      <div class="hero-proof-links">${item.links.map((link) => `<a href="${link.href}">${link.label} ↓</a>`).join("")}</div>
    </article>`).join("");

  const reel = project.showreel;
  document.querySelector("#highlight-reel").innerHTML = `
    <div class="hero-showreel-copy">
      <span class="mono">${reel.eyebrow}</span>
      <h2>${reel.title}</h2>
      <p>${reel.text}</p>
      <small>${reel.meta}</small>
    </div>
    <div class="video-item highlight hero-reel-media" data-method-label="${reel.label}">
      <div class="video-frame">
        <span class="video-label">${reel.label}</span>
        <img class="video-poster" src="${reel.poster}" alt="${reel.posterAlt}" loading="eager" decoding="async" />
        <video data-src="${reel.src}" data-poster="${reel.poster}" preload="none" playsinline hidden aria-hidden="true" aria-label="${reel.label}"></video>
        <button class="video-load" type="button" aria-label="${copy.playVideo}: ${reel.label}">
          <span class="video-play-icon" aria-hidden="true">▶</span>
          <span>${reel.cta}</span>
        </button>
      </div>
      <div class="video-meta"><strong>${reel.label}</strong><span>${reel.duration}</span></div>
    </div>`;
}

function mediaBlock(video, aspect = "landscape") {
  const ready = Boolean(video.src);
  return `
    <div class="video-item ${video.highlight ? "highlight" : ""}" data-method-label="${video.label}">
      <div class="video-frame ${aspect}">
        <span class="video-label">${video.label}</span>
        ${ready
          ? `${video.poster ? `<img class="video-poster" src="${video.poster}" alt="" loading="lazy" decoding="async" />` : ""}
             <video data-src="${video.src}" ${video.poster ? `data-poster="${video.poster}"` : ""} preload="none" playsinline hidden aria-hidden="true" aria-label="${video.label}"></video>
             <button class="video-load" type="button" aria-label="${copy.playVideo}: ${video.label}">
               <span class="video-play-icon" aria-hidden="true">▶</span>
               <span>${copy.playVideo}</span>
             </button>`
          : `<div class="media-placeholder" aria-label="${copy.pendingVideoLabel}"></div>`}
      </div>
      <div class="video-meta"><strong>${video.label}</strong><span>${ready ? copy.playVideo : copy.pending}</span></div>
    </div>`;
}

function bindLazyMedia() {
  document.querySelectorAll(".demo-group").forEach((group) => {
    const inputPoster = group.querySelector('[data-method-label="Input"] .video-poster');
    if (!inputPoster) return;
    const applyAspectRatio = () => {
      if (inputPoster.naturalWidth && inputPoster.naturalHeight) {
        group.style.setProperty("--sample-aspect-ratio", `${inputPoster.naturalWidth} / ${inputPoster.naturalHeight}`);
      }
    };
    if (inputPoster.complete) applyAspectRatio();
    else inputPoster.addEventListener("load", applyAspectRatio, { once: true });
  });

  document.querySelectorAll(".video-load").forEach((button) => {
    button.addEventListener("click", () => {
      const video = button.parentElement.querySelector("video[data-src]");
      if (!video) return;
      button.disabled = true;
      button.querySelector("span:last-child").textContent = copy.loadingVideo;
      if (video.dataset.poster) video.poster = video.dataset.poster;
      video.src = video.dataset.src;
      video.controls = true;
      video.hidden = false;
      video.removeAttribute("aria-hidden");
      video.removeAttribute("data-src");
      video.load();
      const poster = button.parentElement.querySelector(".video-poster");
      if (poster) poster.hidden = true;
      button.hidden = true;
      const status = button.closest(".video-item").querySelector(".video-meta span");
      if (status) status.textContent = copy.ready;
      video.play().catch(() => {});
    }, { once: true });
  });
}

function renderCases(cases) {
  return cases.map((item) => `
    <article id="${item.id}" class="demo-group reveal">
      <div class="demo-head">
        <div>
          ${item.scenario ? `<span class="scenario-label mono">${item.scenario}</span>` : ""}
          <h4>${item.title}</h4>
          <p>${item.description}</p>
        </div>
        <div class="tags">${item.tags.map((tag) => `<span class="tag">${tag}</span>`).join("")}</div>
      </div>
      <div class="method-grid method-grid-${item.methods.length}">${item.methods.map((video) => mediaBlock(video)).join("")}</div>
    </article>`).join("");
}

function renderQualitativeResults() {
  const resultRoot = document.querySelector("#qualitative-results");
  resultRoot.innerHTML = data.resultSettings.map((setting) => {
    const cases = data.qualitativeResults.filter((item) => item.setting === setting.id);
    const casesMarkup = setting.subgroups
      ? `<div class="result-subgroups">${setting.subgroups.map((subgroup) => {
          const subgroupCases = cases.filter((item) => item.subgroup === subgroup.id);
          return `
            <section class="result-subgroup" aria-labelledby="subgroup-${setting.id}-${subgroup.id}">
              <div class="subgroup-head reveal">
                <span class="subgroup-index mono">${subgroup.index}</span>
                <div>
                  <p class="subgroup-kicker">${copy.testCategory}</p>
                  <h4 id="subgroup-${setting.id}-${subgroup.id}">${subgroup.title}${subgroup.titleCn ? `<span>${subgroup.titleCn}</span>` : ""}</h4>
                  <p>${subgroup.description}</p>
                </div>
              </div>
              <div class="result-cases">
                ${subgroupCases.length
                  ? renderCases(subgroupCases)
                  : `<div class="subgroup-empty reveal"><span class="mono">${copy.comingSoon}</span><p>${subgroup.emptyText || copy.emptySamples}</p></div>`}
              </div>
            </section>`;
        }).join("")}</div>`
      : `<div class="result-cases">${renderCases(cases)}</div>`;
    return `
      <section class="result-setting" aria-labelledby="setting-${setting.id}">
        <div class="setting-head reveal">
          <span class="setting-index mono">${setting.index}</span>
          <div>
            <p class="setting-kicker">${copy.evaluationSetting}</p>
            <h3 id="setting-${setting.id}">${setting.title}${setting.titleCn ? `<span>${setting.titleCn}</span>` : ""}</h3>
            <p>${setting.description}</p>
          </div>
        </div>
        ${casesMarkup}
      </section>`;
  }).join("");
  bindLazyMedia();
  observeReveals();
}

function renderMethod() {
  const method = data.method;
  document.querySelector("#method-facts").innerHTML = method.facts.map((fact) => `
    <div class="method-fact">
      <strong>${fact.value}</strong>
      <span>${fact.label}</span>
    </div>`).join("");

  const pipelineImage = document.querySelector("#method-pipeline-image");
  if (pipelineImage) pipelineImage.alt = method.pipelineAlt;

  document.querySelector("#method-details").innerHTML = method.details.map((detail) => `
    <article class="method-detail">
      <div class="method-detail-head">
        <span class="method-detail-index mono">${detail.index}</span>
        <span class="mono">${detail.eyebrow}</span>
      </div>
      <h3>${detail.title}</h3>
      <p>${detail.text}</p>
      ${detail.formula ? `<div class="method-detail-formula mono">${detail.formula}</div>` : ""}
      ${detail.meta ? `<div class="method-detail-meta">${detail.meta.map((item) => `<span>${item}</span>`).join("")}</div>` : ""}
    </article>`).join("");
}

function renderMetrics() {
  const results = data.quantitativeResults;

  document.querySelector("#evaluation-limitation").innerHTML = `
    <span class="limitation-mark" aria-hidden="true">!</span>
    <div>
      <span class="mono">${results.limitation.label}</span>
      <h3>${results.limitation.title}</h3>
      <p>${results.limitation.text}</p>
    </div>`;

  document.querySelector("#metric-cards").innerHTML = results.summary.map((metric) => `
    <article class="metric-card">
      <span>${metric.name}</span>
      <strong>${metric.value}</strong>
      <small>${metric.note}</small>
    </article>`).join("");

  document.querySelector("#metric-notes").innerHTML = results.readingNotes.map((note) => `
    <article class="metric-note">
      <span>${note.eyebrow}</span>
      <h3>${note.title}</h3>
      <p>${note.text}</p>
    </article>`).join("");

  const tableMarkup = (columns, className, caption, rows = results.rows) => {
    const tableHead = columns.map((column) => `
      <th scope="col">
        ${column.group ? `<small>${column.group}</small>` : ""}
        <span>${column.label}</span>
      </th>`).join("");

    const tableRows = rows.map((row) => `
      <tr class="${row.highlight ? "is-ours" : ""} ${row.reference ? "is-reference" : ""}">
        ${columns.map((column, index) => index === 0
          ? `<th scope="row">${row[column.key]}${row.detail ? `<small>${row.detail}</small>` : ""}${row.reference ? `<small>${copy.reference}</small>` : ""}</th>`
          : `<td>${row[column.key]}</td>`).join("")}
      </tr>`).join("");

    return `
      <table class="quantitative-table ${className}">
        <caption>${caption}</caption>
        <thead><tr>${tableHead}</tr></thead>
        <tbody>${tableRows}</tbody>
      </table>`;
  };

  document.querySelector("#quantitative-table").innerHTML = `
    ${tableMarkup(results.primaryColumns, "quantitative-table-primary", copy.tableCaption)}
    <p class="table-footnote">${copy.tableFootnote}</p>`;

  document.querySelector("#subjective-results").innerHTML = `
    <div class="metrics-subhead">
      <span class="mono">${copy.subjectiveEyebrow}</span>
      <h3>${copy.subjectiveTitle}</h3>
      <p>${data.subjectiveResults.intro}</p>
    </div>
    <div class="quantitative-table-wrap">
      ${tableMarkup(data.subjectiveResults.columns, "subjective-table", copy.subjectiveCaption, data.subjectiveResults.rows)}
      <p class="table-footnote">${data.subjectiveResults.note}</p>
    </div>
    <aside class="mos-analysis">
      <span class="mono">MOS ANALYSIS</span>
      <div>
        <h4>${data.subjectiveResults.analysisTitle}</h4>
        <p>${data.subjectiveResults.analysis}</p>
      </div>
    </aside>`;
}

function renderEfficiency() {
  const efficiency = data.efficiency;
  const simpleTable = (columns, rows, caption, className) => `
    <div class="quantitative-table-wrap">
      <table class="quantitative-table ${className}">
        <caption>${caption}</caption>
        <thead><tr>${columns.map((column) => `
          <th scope="col">${column.group ? `<small>${column.group}</small>` : ""}<span>${column.label}</span></th>`).join("")}</tr></thead>
        <tbody>${rows.map((row) => `
          <tr class="${row.highlight ? "is-ours" : ""}">
            ${columns.map((column, index) => index === 0
              ? `<th scope="row">${row[column.key]}${row.detail ? `<small>${row.detail}</small>` : ""}</th>`
              : `<td>${row[column.key]}</td>`).join("")}
          </tr>`).join("")}</tbody>
      </table>
    </div>`;

  document.querySelector("#efficiency-summary").innerHTML = efficiency.summary.map((metric) => `
    <article class="metric-card">
      <span>${metric.name}</span>
      <strong>${metric.value}</strong>
      <small>${metric.note}</small>
    </article>`).join("");

  document.querySelector("#efficiency-tables").innerHTML = `
    <div>
      <h3>${copy.throughputTitle}</h3>
      ${simpleTable(efficiency.throughputColumns, efficiency.throughputRows, copy.throughputCaption, "throughput-table")}
    </div>
    <div>
      <h3>${copy.profileTitle}</h3>
      ${simpleTable(efficiency.profileColumns, efficiency.profileRows, copy.profileCaption, "profile-table")}
    </div>`;

  document.querySelector("#efficiency-scope").innerHTML = `<strong>${copy.timingScope}</strong><p>${efficiency.scope}</p>`;
}

function bindCitation() {
  const code = document.querySelector("#citation-code");
  const button = document.querySelector("#copy-citation");
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(data.citation);
      button.textContent = copy.copied;
      setTimeout(() => { button.textContent = copy.copy; }, 1600);
    } catch {
      button.textContent = copy.selectText;
    }
  });
}

function renderCitation() {
  document.querySelector("#citation-code").textContent = data.citation;
  document.querySelector("#copy-citation").textContent = copy.copy;
}

function renderPage() {
  data = datasets[currentLanguage];
  copy = window.UI_COPY[currentLanguage];
  bindStaticCopy();
  bindProjectContent();
  renderHeroEvidence();
  renderQualitativeResults();
  renderMethod();
  renderMetrics();
  renderEfficiency();
  renderCitation();
  observeReveals();
}

function bindLanguageSwitcher() {
  document.querySelectorAll("[data-language]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextLanguage = button.dataset.language;
      if (nextLanguage === currentLanguage) return;
      currentLanguage = nextLanguage;
      renderPage();
    });
  });
}

let revealObserver;
function observeReveals() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    document.querySelectorAll(".reveal").forEach((element) => element.classList.add("visible"));
    return;
  }
  if (!revealObserver) {
    revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          revealObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });
  }
  document.querySelectorAll(".reveal:not(.visible)").forEach((element) => revealObserver.observe(element));
}

bindCitation();
bindLanguageSwitcher();
renderPage();
