const $ = (id) => document.getElementById(id);  // short for document.getElementById, faster than querySelector

// Guesses and photoUrl are stored so they can be publically accessed
const state = {
  guesses: [],
  photoUrl: null,
};

// This function displays model info and model validation metrics in the page footer
async function showModelInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    const v = info.val || {};
    let text = `Model: ${info.checkpoint} (${info.num_cells} map cells, running on ${info.device})`;
    if (v.geoscore) text += ` · validation GeoScore ${Math.round(v.geoscore)}`;
    if (v.median_km) text += `, median error ${Math.round(v.median_km)} km`;
    $("footer").textContent = text;
  } catch {
    // Empty
  }
}

const drop = $("drop");
const fileInput = $("file");

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) upload(fileInput.files[0]);
  fileInput.value = "";
});

drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); 
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
});

// This function sends the photo to the server and displays the result
async function upload(file) {
  $("status").className = "";
  $("status").textContent = "Predicting…";
  $("warnings").replaceChildren();
  $("result").hidden = true;
  resetCheck();

  const form = new FormData();
  form.append("image", file);

  let data;
  try {
    const response = await fetch("/predict", { method: "POST", body: form });
    data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong.");
  } catch (err) {
    $("status").className = "error";
    $("status").textContent = err.message;
    return;
  }

  showResult(data, file);
}

// This function displays the model's guesses and the user's photo in the page
function showResult(data, file) {
  $("status").textContent = `Done in ${data.elapsed_ms} ms.`;

  for (const text of data.warnings) {
    const box = document.createElement("div");
    box.className = "warning";
    box.textContent = text;
    $("warnings").appendChild(box);
  }

  // Show the user's own photo next to the crop the model actually saw
  if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
  state.photoUrl = URL.createObjectURL(file);
  $("orig").src = state.photoUrl;
  $("crop").src = data.crop;

  state.guesses = data.top;
  const best = data.guess;
  $("best").textContent =
    `Best guess: ${best.lat.toFixed(2)}, ${best.lon.toFixed(2)} (${(100 * best.prob).toFixed(1)}%)`;

  // Making one table row per guess: rank, coordinates, probability bar, percentage, and an empty cell for the distance (filled in later by renderScore)
  const tbody = $("top").querySelector("tbody");
  tbody.replaceChildren();
  data.top.forEach((g, i) => {
    const row = tbody.insertRow();
    row.insertCell().textContent = `${i + 1}.`;
    row.insertCell().textContent = `${g.lat.toFixed(2)}, ${g.lon.toFixed(2)}`;

    const barCell = row.insertCell();
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.style.width = `${100 * g.prob}%`;  // the bar's filled part is as wide as the probability
    bar.appendChild(fill);
    barCell.appendChild(bar);

    row.insertCell().textContent = `${(100 * g.prob).toFixed(1)}%`;
    row.insertCell().className = "dist-col";
  });

  $("result").hidden = false;
}

$("truth-form").addEventListener("submit", checkGuess);

// This function parses the user's typed-in coordinates, returning [lat, lon] or null if invalid
function parseCoordinates(text) {
  if (/[a-z]/i.test(text)) return null;
  const numbers = text.match(/-?(?:\d+\.?\d*|\.\d+)/g);  // every number in the text
  if (!numbers || numbers.length !== 2) return null;
  const [lat, lon] = numbers.map(Number);
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return [lat, lon];
}

// This function sends the user's typed-in coordinates to the server, which computes the distance and GeoScore for each guess, then displays the results
async function checkGuess(event) {
  event.preventDefault();

  const truth = parseCoordinates($("truth").value);
  if (!truth) {
    showScoreMessage("Couldn't read that. Enter two plain numbers, latitude then longitude, " + "like 42.2778, -71.7585 (negative for south or west).", true);
    return;
  }

  // Sends the truth and the model's guesses to the server (which computes the distance and GeoScore for each guess), then returns the results to the page
  let result;
  try {
    const response = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ truth, points: state.guesses.map((g) => [g.lat, g.lon]) }),
    });
    result = await response.json();
    if (!response.ok) throw new Error(result.error || "Something went wrong.");
  } catch (err) {
    showScoreMessage(err.message, true);
    return;
  }

  renderScore(result);
}

// This function displays the distance and GeoScore for each guess, highlighting the guess that was closest to the truth
function renderScore({ distance_km, geoscore }) {
  const km = (d) => Math.round(d).toLocaleString("en-US") + " km";

  // Fill the distance column and mark the guess that landed nearest the truth.
  const rows = $("top").querySelectorAll("tbody tr");
  const closest = distance_km.indexOf(Math.min(...distance_km));
  rows.forEach((row, i) => {
    row.querySelector(".dist-col").textContent = km(distance_km[i]);
    row.classList.toggle("closest", i === closest);
  });
  $("top").classList.add("has-truth");

  // The model's best guess (index 0) is the one that counts as its answer
  const score = $("score");
  score.className = "";
  score.replaceChildren();

  const headline = document.createElement("div");
  headline.className = "big";
  headline.textContent = `${km(distance_km[0])} off · GeoScore ${Math.round(geoscore[0])} / 5000`;
  score.appendChild(headline);

  if (closest !== 0) {
    const note = document.createElement("div");
    note.textContent = `The nearest of the five guesses is #${closest + 1}, ${km(distance_km[closest])} from the truth.`;
    score.appendChild(note);
  }
}

// This function displays a message in the score area, optionally marking it as an error
function showScoreMessage(text, isError) {
  $("score").className = isError ? "error" : "";
  $("score").textContent = text;
}

// This function clears the check panel (used whenever a new photo is uploaded).
function resetCheck() {
  $("truth").value = "";
  $("score").className = "";
  $("score").replaceChildren();
  $("top").classList.remove("has-truth");
  state.guesses = [];
}

showModelInfo();