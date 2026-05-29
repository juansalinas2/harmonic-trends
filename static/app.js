const state = {
  selectedSong: null,
  selectedResult: null,
  results: [],
  searchSpotifyOnly: true,
  resultSpotifyOnly: true,
  excludeSameArtist: false,
  crossGenreOnly: false,
  discoveryPreset: "balanced",
  currentView: "neighbors",
  scales: [3, 4, 5, 6, 7, 8],
  weights: { 3: 1, 4: 1.33, 5: 1.67, 6: 2, 7: 2.33, 8: 2.67 },
  activeScales: new Set([3, 4, 5, 6, 7, 8])
};

const fmt = new Intl.NumberFormat("en-US");
const spotifyMetaCache = new Map();
const defaultWeights = { 3: 1, 4: 1.33, 5: 1.67, 6: 2, 7: 2.33, 8: 2.67 };
const presets = {
  balanced: {
    active: [3, 4, 5, 6, 7, 8],
    weights: defaultWeights
  },
  rare: {
    active: [3, 4, 5, 6, 7, 8],
    weights: { 3: 0.4, 4: 0.7, 5: 1.2, 6: 2.0, 7: 3.2, 8: 4.0 }
  },
  long: {
    active: [6, 7, 8],
    weights: { 3: 0.2, 4: 0.4, 5: 0.8, 6: 1.8, 7: 2.8, 8: 3.8 }
  }
};

function $(id) {
  return document.getElementById(id);
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function compact(value) {
  if (value === null || value === undefined || value === "") return "unknown";
  return String(value);
}

function score(value) {
  if (value === null || value === undefined) return "";
  return Number(value).toFixed(4);
}

function spotifyMode(enabled) {
  return enabled ? "only" : "all";
}

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function selectedNs() {
  return [...state.activeScales].sort((a, b) => a - b);
}

function weightParam() {
  return selectedNs().map((n) => `${n}:${state.weights[n]}`).join(",");
}

function lengthHint(n) {
  return {
    3: "short motifs",
    4: "common turns",
    5: "phrase fragments",
    6: "longer motion",
    7: "distinctive arcs",
    8: "rare fingerprints"
  }[n];
}

function renderScales() {
  $("scaleGrid").innerHTML = state.scales.map((n) => `
    <div class="scale-row ${state.activeScales.has(n) ? "active" : ""}" data-scale-row="${n}">
      <label class="scale-name">
        <input type="checkbox" data-scale-check="${n}" ${state.activeScales.has(n) ? "checked" : ""}>
        <span class="length-chip">H${n}</span>
        <span class="length-copy">
          <strong>${n}-chord patterns</strong>
          <em>${lengthHint(n)}</em>
        </span>
      </label>
      <div class="weight-control">
        <span class="weight-label">Influence</span>
        <input type="range" min="0" max="4" step="0.1" value="${state.weights[n]}" data-scale-weight="${n}" aria-label="H${n} influence weight">
        <span class="weight-value" data-scale-value="${n}">${state.weights[n].toFixed(1)}x</span>
      </div>
    </div>
  `).join("");

  document.querySelectorAll("[data-scale-check]").forEach((input) => {
    input.addEventListener("change", () => {
      const n = Number(input.dataset.scaleCheck);
      if (input.checked) state.activeScales.add(n);
      else state.activeScales.delete(n);
      input.closest("[data-scale-row]").classList.toggle("active", input.checked);
    });
  });

  document.querySelectorAll("[data-scale-weight]").forEach((input) => {
    input.addEventListener("input", () => {
      const n = Number(input.dataset.scaleWeight);
      state.weights[n] = Number(input.value);
      document.querySelector(`[data-scale-value="${n}"]`).textContent = `${state.weights[n].toFixed(1)}x`;
    });
  });
}

function renderStatus(stats) {
  const terms = stats.table_counts.find((row) => row.table_name === "song_harmonic_terms")?.rows;
  const features = stats.table_counts.find((row) => row.table_name === "harmonic_song_document_frequency")?.rows;
  const coverage = stats.coverage.map((row) => `H${row.n} ${(row.weighted_coverage * 100).toFixed(0)}%`).join(" · ");
  $("statusStrip").innerHTML = `
    <span>${fmt.format(terms)} song features</span>
    <span>${fmt.format(features)} harmonic classes</span>
    <span>${coverage}</span>
  `;
}

function updateToggle(button, enabled) {
  button.classList.toggle("active", enabled);
  button.setAttribute("aria-pressed", String(enabled));
}

function setView(view) {
  state.currentView = view;
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${view}View`);
  });
}

function updatePresetButtons() {
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === state.discoveryPreset);
  });
}

function applyPreset(name) {
  const preset = presets[name];
  if (!preset) return;
  state.discoveryPreset = name;
  state.activeScales = new Set(preset.active);
  state.weights = { ...defaultWeights, ...preset.weights };
  renderScales();
  updatePresetButtons();
  runSimilarity();
}

function songLabel(song) {
  return `Song ${esc(song.song_id)}`;
}

function songMetaParts(song, options = {}) {
  const includeSpotify = options.includeSpotify ?? true;
  const artist = song.spotify_artist || song.spotify_artist_name || compact(song.artist_id);
  return [
    compact(song.main_genre),
    song.release_year || song.decade,
    artist,
    includeSpotify && song.spotify_song_id ? `Spotify ${song.spotify_song_id}` : null
  ].filter(Boolean);
}

function songMeta(song, options = {}) {
  return esc(songMetaParts(song, options).join(" · "));
}

function resultMeta(song) {
  return songMeta(song, { includeSpotify: false });
}

function displayName(song) {
  const id = song.song_id ?? song.candidate_song_id;
  return song.spotify_title || `Song ${id}`;
}

function spotifyLink(song) {
  if (!song.spotify_song_id) return "";
  return `https://open.spotify.com/track/${encodeURIComponent(song.spotify_song_id)}`;
}

function spotifyEmbed(song) {
  if (!song.spotify_song_id) return "";
  return `https://open.spotify.com/embed/track/${encodeURIComponent(song.spotify_song_id)}`;
}

async function spotifyMeta(spotifySongId) {
  if (!spotifySongId) return null;
  if (spotifyMetaCache.has(spotifySongId)) return spotifyMetaCache.get(spotifySongId);

  const trackUrl = `https://open.spotify.com/track/${encodeURIComponent(spotifySongId)}`;
  const oembedUrl = `https://open.spotify.com/oembed?url=${encodeURIComponent(trackUrl)}`;
  const promise = (async () => {
    try {
      const payload = await api("/api/spotify", { spotify_song_id: spotifySongId });
      if (payload.title || payload.artist || payload.author_name) {
        return {
          title: payload.title || null,
          artist: payload.artist || payload.author_name || null
        };
      }
    } catch {
      // Fall back to direct oEmbed if the local metadata proxy cannot reach Spotify.
    }

    try {
      const response = await fetch(oembedUrl);
      if (response.ok) {
        const payload = await response.json();
        return {
          title: payload.title || null,
          artist: payload.author_name || null
        };
      }
    } catch {
      // Browser-side Spotify oEmbed is best-effort; fall through to local API.
    }

    return { title: null, artist: null };
  })();

  spotifyMetaCache.set(spotifySongId, promise);
  return promise;
}

function spotifyLabelMarkup(song, options = {}) {
  const id = song.song_id ?? song.candidate_song_id;
  const fallback = song.spotify_title || options.fallbackTitle || `Song ${id}`;
  const cachedArtist = song.spotify_artist || song.spotify_artist_name || "";
  const prefix = options.prefix || "";
  const title = `${prefix}${esc(fallback)}`;
  if (!song.spotify_song_id) {
    return `
      <span class="spotify-label ${cachedArtist ? "" : "no-artist"}">
        <span class="label-title">${title}</span>
        <span class="song-artist" ${cachedArtist ? "" : "hidden"}>${esc(cachedArtist)}</span>
      </span>
    `;
  }
  return `
    <span class="spotify-label ${cachedArtist ? "" : "no-artist"}" data-spotify-label="${esc(song.spotify_song_id)}" data-fallback-title="${esc(fallback)}" data-prefix="${esc(prefix)}">
      <span class="label-title"><span data-prefix>${prefix}</span><span data-title>${esc(fallback)}</span></span>
      <span data-artist class="song-artist" ${cachedArtist ? "" : "hidden"}>${esc(cachedArtist)}</span>
    </span>
  `;
}

function renderFocusBar() {
  if (!state.selectedSong) {
    $("focusBar").innerHTML = `<div class="empty-state">Search for a recognizable seed song.</div>`;
    return;
  }

  const result = state.selectedResult;
  $("focusBar").innerHTML = `
    <div class="focus-card">
      <span class="focus-kicker">Seed</span>
      <strong>${spotifyLabelMarkup(state.selectedSong, { fallbackTitle: `Song ${state.selectedSong.song_id}` })}</strong>
      <span class="song-meta">${songMeta(state.selectedSong, { includeSpotify: false })}</span>
    </div>
    <div class="focus-arrow">→</div>
    <div class="focus-card ${result ? "" : "muted-card"}">
      <span class="focus-kicker">Selected Neighbor</span>
      ${result ? `
        <strong>${spotifyLabelMarkup(result, { fallbackTitle: `Song ${result.candidate_song_id}` })}</strong>
        <span class="song-meta">${score(result.similarity_score)} score · ${fmt.format(result.shared_features)} shared</span>
      ` : `<span class="song-meta">Run a query or select a neighbor.</span>`}
    </div>
  `;
  hydrateSpotifyLabels($("focusBar"));
}

async function hydrateSpotifyLabels(container = document) {
  const targets = [...container.querySelectorAll("[data-spotify-label]")];
  await Promise.all(targets.map(async (target) => {
    const spotifySongId = target.dataset.spotifyLabel;
    const meta = await spotifyMeta(spotifySongId);
    const title = meta?.title || target.dataset.fallbackTitle || "Spotify track";
    const artist = meta?.artist || "";
    target.querySelector("[data-title]").textContent = title;
    const prefixEl = target.querySelector("[data-prefix]");
    if (prefixEl) prefixEl.textContent = target.dataset.prefix || "";
    const artistEl = target.querySelector("[data-artist]");
    if (artistEl) {
      artistEl.textContent = artist;
      artistEl.hidden = !artist;
    }
    target.classList.toggle("no-artist", !artist);
  }));
}

function renderSearchResults(results) {
  if (!results.length) {
    const suffix = state.searchSpotifyOnly ? " Try turning off w/ Spotify ID." : "";
    $("searchResults").innerHTML = `<div class="empty-state">No indexed songs found.${suffix}</div>`;
    return;
  }

  $("searchResults").innerHTML = results.map((song) => `
    <button class="song-option ${state.selectedSong?.song_id === song.song_id ? "active" : ""}" data-song-id="${esc(song.song_id)}" type="button">
      <span class="song-option-main">
        <span class="song-title">${spotifyLabelMarkup(song, { fallbackTitle: `Song ${song.song_id}` })}</span>
        ${song.spotify_song_id ? `<span class="spotify-badge">Spotify</span>` : ""}
      </span>
      <span class="song-meta">${resultMeta(song)}</span>
      <span class="song-stats">${fmt.format(song.indexed_features || 0)} features · ${fmt.format(song.indexed_windows || 0)} windows</span>
    </button>
  `).join("");

  hydrateSpotifyLabels($("searchResults"));

  document.querySelectorAll("[data-song-id]").forEach((button) => {
    button.addEventListener("click", () => selectSong(Number(button.dataset.songId)));
  });
}

function renderSelectedSong(payload) {
  const song = payload.song;
  const totals = payload.totals || [];
  const coverage = totals.map((row) => `H${row.n} ${(row.indexed_window_coverage * 100).toFixed(0)}%`).join(" · ");
  $("selectedSong").innerHTML = `
    <div class="song-title">${spotifyLabelMarkup(song, { fallbackTitle: `Song ${song.song_id}` })}</div>
    <div class="song-meta">${songMeta(song)}</div>
    <div class="song-meta">${coverage}</div>
  `;
  hydrateSpotifyLabels($("selectedSong"));
  renderFocusBar();
  renderPreview(song);
}

function previewMarkup(song, label) {
  if (song.spotify_song_id) {
    return `
      <div class="preview-title">${esc(label)}</div>
      <iframe
        title="Spotify preview for Song ${esc(song.song_id)}"
        src="${spotifyEmbed(song)}"
        width="100%"
        height="80"
        frameborder="0"
        scrolling="no"
        style="overflow:hidden"
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
        loading="lazy"></iframe>
      <div class="preview-actions">
        <a href="${spotifyLink(song)}" target="_blank" rel="noreferrer">Spotify</a>
        <span data-spotify-name="${esc(song.spotify_song_id)}" class="song-meta">Loading Spotify title...</span>
      </div>
    `;
  }

  return `
    <div class="preview-title">${esc(label)}</div>
    <div class="song-meta">No Spotify track id is available in the local metadata.</div>
    <div class="chord-excerpt">${esc(song.chord_excerpt || "No chord excerpt found.")}</div>
  `;
}

async function hydrateSpotifyNames(container) {
  const targets = [...container.querySelectorAll("[data-spotify-name]")];
  await Promise.all(targets.map(async (target) => {
    const spotifySongId = target.dataset.spotifyName;
    const meta = await spotifyMeta(spotifySongId);
    target.textContent = [meta?.title, meta?.artist].filter(Boolean).join(" · ") || "Spotify title unavailable";
  }));
}

async function renderPreview(song) {
  $("previewPanel").innerHTML = previewMarkup(song, "Selected Song Preview");
  await hydrateSpotifyNames($("previewPanel"));
}

async function renderCandidatePreview(song) {
  $("candidatePreview").innerHTML = previewMarkup(song, "Selected Result Preview");
  await hydrateSpotifyNames($("candidatePreview"));
}

async function search() {
  const q = $("searchInput").value.trim();
  $("searchResults").innerHTML = `<div class="empty-state">Searching...</div>`;
  const payload = await api("/api/search", {
    q,
    limit: 20,
    spotify: spotifyMode(state.searchSpotifyOnly)
  });
  renderSearchResults(payload.results);
}

async function selectSong(songId) {
  const payload = await api("/api/song", { song_id: songId });
  state.selectedSong = payload.song;
  state.selectedResult = null;
  renderFocusBar();
  renderSelectedSong(payload);
  await runSimilarity();
}

function renderResults(results) {
  state.results = results;
  if (!results.length) {
    const suffix = state.resultSpotifyOnly ? " Try turning off Neighbors w/ Spotify ID." : "";
    $("resultsTable").innerHTML = `<div class="empty-state">No candidates matched the current scale and shared-feature settings.${suffix}</div>`;
    return;
  }

  $("resultsTable").innerHTML = `
    <div class="neighbor-list">
      ${results.map((row, index) => `
        <button class="neighbor-card ${state.selectedResult?.candidate_song_id === row.candidate_song_id ? "active" : ""}" data-candidate-id="${esc(row.candidate_song_id)}" type="button">
          <span class="neighbor-rank">${index + 1}</span>
          <span class="neighbor-main">
            <strong>${spotifyLabelMarkup(row, { fallbackTitle: `Song ${row.candidate_song_id}` })}</strong>
            <span class="song-meta result-submeta">${resultMeta(row)}</span>
            ${row.spotify_song_id ? `<span class="track-badge">Spotify track</span>` : ""}
          </span>
          <span class="neighbor-stats">
            <span><strong>${score(row.similarity_score)}</strong><em>score</em></span>
            <span><strong>${fmt.format(row.shared_features)}</strong><em>shared</em></span>
          </span>
        </button>
      `).join("")}
    </div>
  `;

  hydrateSpotifyLabels($("resultsTable"));

  document.querySelectorAll("[data-candidate-id]").forEach((row) => {
    row.addEventListener("click", () => {
      const candidateId = Number(row.dataset.candidateId);
      const result = state.results.find((item) => item.candidate_song_id === candidateId);
      selectResult(result, true);
    });
  });
  renderFocusBar();
}

async function runSimilarity() {
  if (!state.selectedSong) return;
  if (!selectedNs().length) {
    $("resultsTable").innerHTML = `<div class="empty-state">Select at least one chord pattern length.</div>`;
    return;
  }

  $("resultsTable").innerHTML = `<div class="empty-state">Computing harmonic neighbors...</div>`;
  const payload = await api("/api/similar", {
    song_id: state.selectedSong.song_id,
    ns: selectedNs().join(","),
    weights: weightParam(),
    top_k: $("topKInput").value,
    min_shared: $("minSharedInput").value,
    spotify: spotifyMode(state.resultSpotifyOnly),
    exclude_same_artist: state.excludeSameArtist ? "1" : "0",
    cross_genre: state.crossGenreOnly ? "1" : "0"
  });
  renderResults(payload.results);
  loadComparison();
  if (payload.results.length) selectResult(payload.results[0]);
}

async function loadComparison() {
  if (!state.selectedSong) return;
  $("comparisonPanel").innerHTML = `<div class="empty-state">Comparing fixed-n rankings...</div>`;
  const payload = await api("/api/compare-n", {
    song_id: state.selectedSong.song_id,
    top_k: 3,
    min_shared: Math.min(Number($("minSharedInput").value || 4), 4),
    spotify: spotifyMode(state.resultSpotifyOnly),
    exclude_same_artist: state.excludeSameArtist ? "1" : "0",
    cross_genre: state.crossGenreOnly ? "1" : "0"
  });
  renderComparison(payload.results_by_n);
}

function renderComparison(resultsByN) {
  $("comparisonPanel").innerHTML = `
    <div class="comparison-header">
      <span>Pattern length check</span>
      <span>One length at a time</span>
    </div>
    <div class="comparison-grid">
      ${state.scales.map((n) => {
        const rows = resultsByN[String(n)] || [];
        return `
          <div class="comparison-column">
            <h3>H${n} · ${n}-chord only</h3>
            ${rows.length ? rows.map((row, index) => `
              <div class="comparison-item">
                <strong>${spotifyLabelMarkup(row, { fallbackTitle: `Song ${row.candidate_song_id}`, prefix: `${index + 1}. ` })}</strong>
                <span class="song-meta">${score(row.similarity_score)} · ${fmt.format(row.shared_features)} shared</span>
              </div>
            `).join("") : `<div class="comparison-item"><span class="song-meta">No match</span></div>`}
          </div>
        `;
      }).join("")}
    </div>
  `;
  hydrateSpotifyLabels($("comparisonPanel"));
}

async function selectResult(result, openEvidence = false) {
  if (!state.selectedSong || !result) return;
  state.selectedResult = result;
  renderResults(state.results);
  renderFocusBar();
  if (openEvidence) setView("evidence");
  $("pairLabel").textContent = `Song ${state.selectedSong.song_id} -> Song ${result.candidate_song_id}`;
  $("candidatePreview").innerHTML = `<div class="empty-state">Loading selected result preview...</div>`;
  $("evidenceSummary").innerHTML = "";
  $("evidenceTable").innerHTML = `<div class="empty-state">Loading harmonic evidence...</div>`;
  const [payload, candidatePayload] = await Promise.all([
    api("/api/explain", {
      song_id: state.selectedSong.song_id,
      candidate_id: result.candidate_song_id,
      ns: selectedNs().join(","),
      weights: weightParam(),
      top_features: 50
    }),
    api("/api/song", { song_id: result.candidate_song_id })
  ]);
  renderEvidenceSummary(payload.by_n, payload.features, result);
  renderBreakdown(payload.by_n);
  renderEvidence(payload.features, payload.unique_query);
  await renderCandidatePreview(candidatePayload.song);
}

function renderEvidenceSummary(byN, features, result) {
  const topN = [...byN].sort((a, b) => (b.cosine_contribution || 0) - (a.cosine_contribution || 0))[0];
  const rareFeatures = features.filter((row) => Number(row.song_df) <= 100);
  const rarest = features.reduce((best, row) => {
    if (!best || Number(row.song_df) < Number(best.song_df)) return row;
    return best;
  }, null);
  const longRows = byN.filter((row) => Number(row.n) >= 6);
  const longShared = longRows.reduce((sum, row) => sum + Number(row.shared_features || 0), 0);
  const longContribution = longRows.reduce((sum, row) => sum + Number(row.cosine_contribution || 0), 0);
  const totalContribution = byN.reduce((sum, row) => sum + Number(row.cosine_contribution || 0), 0);
  const longPct = totalContribution ? Math.round((longContribution / totalContribution) * 100) : 0;

  $("evidenceSummary").innerHTML = `
    <div class="summary-card blue">
      <span class="summary-label">Strongest length</span>
      <strong>${topN ? `H${topN.n}` : "n/a"}</strong>
      <span>${topN ? `${fmt.format(topN.shared_features)} shared patterns` : "No contribution"}</span>
    </div>
    <div class="summary-card violet">
      <span class="summary-label">Rare matches</span>
      <strong>${fmt.format(rareFeatures.length)}</strong>
      <span>${rarest ? `rarest appears in ${fmt.format(rarest.song_df)} songs` : "none in top features"}</span>
    </div>
    <div class="summary-card gold">
      <span class="summary-label">Long-pattern evidence</span>
      <strong>${longPct}%</strong>
      <span>${fmt.format(longShared)} shared H6-H8 patterns</span>
    </div>
    <div class="summary-card green">
      <span class="summary-label">Rank signal</span>
      <strong>${score(result.similarity_score)}</strong>
      <span>${fmt.format(result.shared_features)} total shared patterns</span>
    </div>
  `;
}

function renderBreakdown(rows) {
  if (!rows.length) {
    $("breakdown").innerHTML = "";
    return;
  }
  const maxValue = Math.max(...rows.map((row) => row.cosine_contribution || 0), 0.0001);
  $("breakdown").innerHTML = rows.map((row) => {
    const height = Math.max(2, ((row.cosine_contribution || 0) / maxValue) * 100);
    return `
      <div class="bar-cell">
        <div class="bar-label"><span>H${row.n}</span><span>${score(row.cosine_contribution)}</span></div>
        <div class="bar-track n-${row.n}"><div class="bar-fill" style="height:${height}%"></div></div>
        <div class="song-meta">${fmt.format(row.shared_features)} shared</div>
      </div>
    `;
  }).join("");
}

function renderEvidence(features, uniqueQuery = []) {
  if (!features.length) {
    $("evidenceTable").innerHTML = `<div class="empty-state">No shared harmonic features under the current settings.</div>`;
    return;
  }
  $("evidenceTable").innerHTML = `
    <div class="evidence-section">
      <h3>Shared Characteristic Patterns</h3>
    <table>
      <thead>
        <tr>
          <th>n</th>
          <th>Shared harmonic pattern</th>
          <th>Contribution</th>
          <th>IDF</th>
          <th>Song df</th>
          <th>Counts</th>
        </tr>
      </thead>
      <tbody>
        ${features.map((row) => `
          <tr>
            <td>H${row.n}</td>
            <td class="progression">${esc(row.example_ngram)}</td>
            <td class="score">${score(row.cosine_contribution)}</td>
            <td>${Number(row.idf).toFixed(2)}</td>
            <td>${fmt.format(row.song_df)}</td>
            <td>${row.query_count} / ${row.candidate_count}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
    </div>
    <div class="evidence-section">
      <h3>Strong Query-Only Patterns</h3>
      ${uniqueQuery.length ? `
        <table>
          <thead>
            <tr>
              <th>n</th>
              <th>Unique harmonic pattern</th>
              <th>Weight</th>
              <th>IDF</th>
              <th>Song df</th>
              <th>Count</th>
            </tr>
          </thead>
          <tbody>
            ${uniqueQuery.map((row) => `
              <tr>
                <td>H${row.n}</td>
                <td class="progression">${esc(row.example_ngram)}</td>
                <td class="score">${score(row.feature_weight)}</td>
                <td>${Number(row.idf).toFixed(2)}</td>
                <td>${fmt.format(row.song_df)}</td>
                <td>${row.count}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      ` : `<div class="empty-state">No strong query-only patterns under the current settings.</div>`}
    </div>
  `;
}

async function init() {
  renderScales();
  updateToggle($("spotifySearchToggle"), state.searchSpotifyOnly);
  updateToggle($("spotifyResultsToggle"), state.resultSpotifyOnly);
  updateToggle($("excludeArtistToggle"), state.excludeSameArtist);
  updateToggle($("crossGenreToggle"), state.crossGenreOnly);
  updatePresetButtons();
  renderFocusBar();

  $("searchButton").addEventListener("click", search);
  $("searchInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") search();
  });
  $("runButton").addEventListener("click", runSimilarity);
  $("spotifySearchToggle").addEventListener("click", () => {
    state.searchSpotifyOnly = !state.searchSpotifyOnly;
    updateToggle($("spotifySearchToggle"), state.searchSpotifyOnly);
    search();
  });
  $("spotifyResultsToggle").addEventListener("click", () => {
    state.resultSpotifyOnly = !state.resultSpotifyOnly;
    updateToggle($("spotifyResultsToggle"), state.resultSpotifyOnly);
    runSimilarity();
  });
  $("excludeArtistToggle").addEventListener("click", () => {
    state.excludeSameArtist = !state.excludeSameArtist;
    updateToggle($("excludeArtistToggle"), state.excludeSameArtist);
    runSimilarity();
  });
  $("crossGenreToggle").addEventListener("click", () => {
    state.crossGenreOnly = !state.crossGenreOnly;
    updateToggle($("crossGenreToggle"), state.crossGenreOnly);
    runSimilarity();
  });
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
  });
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  $("allScalesButton").addEventListener("click", () => {
    state.activeScales = new Set(state.scales);
    renderScales();
  });

  const stats = await api("/api/stats");
  renderStatus(stats);
  const payload = await api("/api/search", {
    q: "",
    limit: 12,
    spotify: spotifyMode(state.searchSpotifyOnly)
  });
  renderSearchResults(payload.results);
}

init().catch((error) => {
  $("statusStrip").innerHTML = `<span>${esc(error.message)}</span>`;
  console.error(error);
});
