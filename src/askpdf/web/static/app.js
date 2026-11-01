// askpdf: one page, no build step. Model and PDF text is only ever inserted with
// textContent, never as HTML (design §9).
"use strict";

const $ = (id) => document.getElementById(id);
let conversation = null;
let conversationReady = null; // the first conversation is being started

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function pageRef(page, label) {
  return label === String(page) ? `page ${page}` : `p. ${label} (PDF page ${page})`;
}

function link(text, href) {
  if (!href) return el("span", "", text);
  const a = el("a", "", text);
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener";
  return a;
}

// Server-sent events from a fetch() body: EventSource cannot POST.
async function* events(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let name = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      yield { name, data: JSON.parse(data) };
    }
  }
}

async function post(path, body) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
}

async function newConversation() {
  const response = await post("/api/conversations");
  conversation = (await response.json()).id;
}

// --- documents and indexing progress -------------------------------------------------------

async function loadDocs() {
  const data = await (await fetch("/api/docs")).json();
  $("folder").textContent = data.folder;
  const docs = $("docs");
  docs.replaceChildren();
  for (const doc of data.documents) {
    const item = el("li");
    item.append(link(doc.path, `/pdf/${doc.id}`));
    item.append(el("span", "muted", ` · ${doc.pages} ${doc.pages === 1 ? "page" : "pages"}`));
    if (doc.scanned) item.append(el("span", "muted", " · looks scanned, no text"));
    docs.append(item);
  }
  const failed = $("failed");
  failed.replaceChildren();
  for (const f of data.failed) failed.append(el("li", "", `${f.path}: ${f.reason}`));
  const n = data.documents.length;
  $("doc-count").textContent = `(${n})`;
  return n;
}

function watchIndexing() {
  const source = new EventSource("/api/events");
  source.addEventListener("progress", async (event) => {
    const p = JSON.parse(event.data);
    if (p.state === "indexing") {
      $("status").textContent = p.current
        ? `Indexing ${p.current}… (${p.done} read)`
        : "Checking the folder…";
      return;
    }
    source.close();
    const n = await loadDocs();
    $("status").textContent =
      p.state === "failed"
        ? `Indexing failed: ${p.message}`
        : `${n} ${n === 1 ? "document" : "documents"} ready`;
  });
  source.onerror = () => {
    source.close();
    $("status").textContent = "Lost contact with askpdf; is it still running?";
  };
}

// --- one question and its answer -----------------------------------------------------------

function showPages(turn, pages) {
  const box = turn.querySelector(".pages");
  box.querySelector("summary").textContent =
    pages.length === 0
      ? "No page matched the question"
      : `Reading ${pages.length} ${pages.length === 1 ? "page" : "pages"}`;
  const list = box.querySelector("ol");
  for (const p of pages) {
    const item = el("li");
    item.append(link(`${p.file}, ${pageRef(p.page, p.label)}`, p.href));
    item.append(el("span", "snippet", p.snippet));
    list.append(item);
  }
  box.hidden = false;
}

// The answer text, with each [n] turned into a link to its page.
function answerText(text, citations) {
  const byNumber = new Map(citations.map((c) => [c.number, c]));
  const para = el("p", "answer");
  let last = 0;
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    para.append(text.slice(last, match.index));
    const c = byNumber.get(Number(match[1]));
    const a = link(match[0], c && c.href);
    a.classList.add("cite");
    if (c) a.title = `${c.file}, ${pageRef(c.page, c.label)}`;
    para.append(a);
    last = match.index + match[0].length;
  }
  para.append(text.slice(last));
  return para;
}

function showFinal(turn, final) {
  const result = turn.querySelector(".result");
  turn.querySelector(".raw").hidden = true;
  result.replaceChildren();
  if (final.not_found) {
    result.append(el("p", "notice", "The answer is not in these documents."));
    result.hidden = false;
    return;
  }
  result.append(answerText(final.text, final.citations));
  result.append(
    final.grounded
      ? el("span", "verdict ok", "Quoted from the cited pages")
      : el("span", "verdict bad", "Ungrounded: no quote was found on its cited page; check it"),
  );
  const sources = el("ol", "sources");
  for (const c of final.citations) {
    const item = el("li");
    item.append(`[${c.number}] `, link(`${c.file}, ${pageRef(c.page, c.label)}`, c.href));
    if (c.quotes.length === 0) item.append(el("div", "note", "No quote given"));
    if (c.quote_on) {
      const note = el("div", "note", "The quoted words are on ");
      note.append(link(`${c.quote_on.file}, ${pageRef(c.quote_on.page, c.quote_on.label)}`, c.quote_on.href));
      note.append("; the answer numbered its source wrongly.");
      item.append(note);
    }
    const quotes = el("ul", "quotes");
    for (const q of c.quotes) {
      const line = el("li");
      line.append(
        q.verified ? el("span", "ok", "✓ ") : el("span", "bad", `✗ ${q.reason}: `),
        `“${q.quote}”`,
      );
      quotes.append(line);
    }
    item.append(quotes);
    sources.append(item);
  }
  result.append(sources);
  if (final.dropped.length) {
    result.append(
      el("p", "note", `Removed citations of pages the model was not given: ${final.dropped.join(", ")}`),
    );
  }
  result.hidden = false;
}

function showError(turn, message) {
  const result = turn.querySelector(".result");
  result.replaceChildren(el("p", "error", message));
  result.hidden = false;
}

async function ask(question) {
  $("empty")?.remove();
  const turn = $("turn").content.firstElementChild.cloneNode(true);
  turn.querySelector(".question").textContent = question;
  $("log").append(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "start" });

  const waiting = turn.querySelector(".waiting");
  const raw = turn.querySelector(".raw");
  const started = Date.now();
  const tick = () => {
    waiting.textContent = `Reading the pages… ${Math.round((Date.now() - started) / 1000)} s`;
  };
  tick();
  waiting.hidden = false;
  const timer = setInterval(tick, 1000);
  const stopWaiting = () => {
    clearInterval(timer);
    waiting.hidden = true;
  };

  try {
    await conversationReady;
    let response = await post("/api/ask", { conversation_id: conversation, question });
    if (response.status === 404) {
      // askpdf was restarted and forgot the conversation: start a new one.
      await newConversation();
      response = await post("/api/ask", { conversation_id: conversation, question });
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const reason = Array.isArray(detail.detail) ? detail.detail[0]?.msg : detail.detail;
      throw new Error(reason || `askpdf answered ${response.status}`);
    }
    for await (const { name, data } of events(response)) {
      if (name === "search") {
        const note = turn.querySelector(".searched");
        note.textContent = `Searched for: ${data.query}`;
        note.hidden = false;
      } else if (name === "pages") {
        showPages(turn, data.pages);
      } else if (name === "token") {
        stopWaiting();
        raw.hidden = false;
        raw.textContent += data.text;
      } else if (name === "final") {
        stopWaiting();
        showFinal(turn, data);
      } else if (name === "error") {
        stopWaiting();
        showError(turn, data.message);
      }
    }
  } catch (error) {
    showError(turn, `Something went wrong: ${error.message}`);
  } finally {
    stopWaiting();
  }
}

// --- wiring --------------------------------------------------------------------------------

$("ask").addEventListener("submit", async (event) => {
  event.preventDefault();
  const box = $("question");
  const question = box.value.trim();
  if (!question) return;
  box.value = "";
  $("send").disabled = true;
  try {
    await ask(question);
  } finally {
    $("send").disabled = false;
    box.focus();
  }
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("ask").requestSubmit();
  }
});

$("new-chat").addEventListener("click", async () => {
  conversationReady = newConversation();
  await conversationReady;
  $("log").replaceChildren();
  $("question").focus();
});

conversationReady = newConversation();
loadDocs().catch(() => {});
watchIndexing();
