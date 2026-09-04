const video = document.querySelector("#video");
const uploadedImage = document.querySelector("#uploaded-image");
const overlay = document.querySelector("#overlay");
const capture = document.querySelector("#capture");
const cameraButton = document.querySelector("#camera-button");
const stopButton = document.querySelector("#stop-button");
const fileInput = document.querySelector("#file-input");
const emptyState = document.querySelector("#empty-state");
const errorBox = document.querySelector("#error");
const probabilities = document.querySelector("#probabilities");
const primaryEmotion = document.querySelector("#primary-emotion");
const faceCount = document.querySelector("#face-count");
const latency = document.querySelector("#latency");
const modeLabel = document.querySelector("#mode-label");

let stream = null;
let socket = null;
let frameTimer = null;
let frameId = 0;
let waitingForResult = false;
const smoothed = new Map();

function showError(message = "") { errorBox.textContent = message; }

function drawResults(result, sourceWidth, sourceHeight) {
  const rect = overlay.getBoundingClientRect();
  overlay.width = Math.round(rect.width * devicePixelRatio);
  overlay.height = Math.round(rect.height * devicePixelRatio);
  const ctx = overlay.getContext("2d");
  ctx.scale(devicePixelRatio, devicePixelRatio);

  const fit = Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
  const drawWidth = sourceWidth * fit;
  const drawHeight = sourceHeight * fit;
  const offsetX = (rect.width - drawWidth) / 2;
  const offsetY = (rect.height - drawHeight) / 2;
  ctx.lineWidth = 2;
  ctx.font = "600 13px Inter, sans-serif";

  result.faces.forEach((face) => {
    const x = offsetX + face.box.x * fit;
    const y = offsetY + face.box.y * fit;
    const width = face.box.width * fit;
    const height = face.box.height * fit;
    ctx.strokeStyle = "#d9ff58";
    ctx.strokeRect(x, y, width, height);
    const label = `${face.emotion} ${Math.round(face.confidence * 100)}%`;
    const labelWidth = ctx.measureText(label).width + 14;
    ctx.fillStyle = "#d9ff58";
    ctx.fillRect(x, Math.max(0, y - 25), labelWidth, 25);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 7, Math.max(17, y - 8));
  });
}

function updateInsights(result) {
  latency.textContent = `${result.latency_ms} ms`;
  faceCount.textContent = `${result.faces.length} face${result.faces.length === 1 ? "" : "s"} detected`;
  if (!result.faces.length) {
    primaryEmotion.textContent = "No face detected";
    probabilities.replaceChildren();
    return;
  }
  const face = result.faces[0];
  primaryEmotion.textContent = face.emotion;
  Object.entries(face.probabilities).forEach(([emotion, score]) => {
    const previous = smoothed.get(emotion) ?? score;
    smoothed.set(emotion, previous * 0.6 + score * 0.4);
  });
  const ordered = [...smoothed.entries()].sort((a, b) => b[1] - a[1]);
  probabilities.replaceChildren(...ordered.map(([emotion, score]) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `<span>${emotion}</span><span class="bar"><span style="width:${score * 100}%"></span></span><span>${Math.round(score * 100)}%</span>`;
    return row;
  }));
}

function stopCamera() {
  if (frameTimer) clearInterval(frameTimer);
  frameTimer = null;
  if (socket) socket.close();
  socket = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  video.style.display = "none";
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  emptyState.style.display = "grid";
  cameraButton.disabled = false;
  stopButton.disabled = true;
  modeLabel.textContent = "IDLE";
  waitingForResult = false;
}

function connectLiveSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/api/live`);
  socket.addEventListener("message", (event) => {
    waitingForResult = false;
    const result = JSON.parse(event.data);
    if (result.error) return showError(result.error);
    showError();
    updateInsights(result);
    drawResults(result, result.image.width, result.image.height);
  });
  socket.addEventListener("close", () => { waitingForResult = false; });
  socket.addEventListener("error", () => showError("Live inference connection failed."));
}

cameraButton.addEventListener("click", async () => {
  try {
    showError();
    uploadedImage.style.display = "none";
    video.style.display = "block";
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    emptyState.style.display = "none";
    cameraButton.disabled = true;
    stopButton.disabled = false;
    modeLabel.textContent = "LIVE CAMERA";
    connectLiveSocket();
    frameTimer = setInterval(() => {
      if (!socket || socket.readyState !== WebSocket.OPEN || waitingForResult || !video.videoWidth) return;
      const width = 640;
      const height = Math.round(width * video.videoHeight / video.videoWidth);
      capture.width = width;
      capture.height = height;
      capture.getContext("2d").drawImage(video, 0, 0, width, height);
      waitingForResult = true;
      socket.send(JSON.stringify({ frame_id: ++frameId, image: capture.toDataURL("image/jpeg", 0.72) }));
    }, 250);
  } catch (error) {
    stopCamera();
    showError(error.name === "NotAllowedError" ? "Camera permission was denied." : "The laptop camera could not be started.");
  }
});

stopButton.addEventListener("click", stopCamera);

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  stopCamera();
  emptyState.style.display = "none";
  uploadedImage.src = URL.createObjectURL(file);
  uploadedImage.style.display = "block";
  modeLabel.textContent = "IMAGE";
  try {
    const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": file.type }, body: file });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Image analysis failed");
    await uploadedImage.decode();
    updateInsights(result);
    drawResults(result, result.image.width, result.image.height);
    showError();
  } catch (error) {
    showError(error.message);
  } finally {
    fileInput.value = "";
  }
});

window.addEventListener("resize", () => overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height));

