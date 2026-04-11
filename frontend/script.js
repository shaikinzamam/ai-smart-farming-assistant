const API_BASE_URL = "http://localhost:8000";

const plantImageInput = document.getElementById("plantImage");
const previewBox = document.getElementById("previewBox");
const predictBtn = document.getElementById("predictBtn");
const diseaseResult = document.getElementById("diseaseResult");
const organicSolution = document.getElementById("organicSolution");
const chemicalSolution = document.getElementById("chemicalSolution");
const confidenceResult = document.getElementById("confidenceResult");
const apiStatus = document.getElementById("apiStatus");

const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");
const voiceBtn = document.getElementById("voiceBtn");
const muteBtn = document.getElementById("muteBtn");

let selectedFile = null;
let voiceReplyEnabled = true;

plantImageInput.addEventListener("change", handleImageSelection);
predictBtn.addEventListener("click", predictDisease);
chatForm.addEventListener("submit", handleChatSubmit);
voiceBtn.addEventListener("click", startVoiceInput);
muteBtn.addEventListener("click", toggleVoiceReply);

checkBackendStatus();

function handleImageSelection(event) {
  selectedFile = event.target.files[0] || null;

  if (!selectedFile) {
    previewBox.innerHTML = "<p>No image selected</p>";
    return;
  }

  const previewUrl = URL.createObjectURL(selectedFile);
  previewBox.innerHTML = `<img src="${previewUrl}" alt="Plant preview" />`;
}

async function predictDisease() {
  if (!selectedFile) {
    updateDiseaseResult({
      disease: "No image selected",
      organic_solution: "Please upload a plant image before starting prediction.",
      chemical_solution: "No treatment recommendation is available yet.",
      confidence: "Demo estimate pending",
    });
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    updateDiseaseResult(data);
    apiStatus.textContent = "Backend: connected";
  } catch (error) {
    updateDiseaseResult({
      disease: "Prediction failed",
      organic_solution: error.message,
      chemical_solution: "Make sure the FastAPI backend is running on localhost:8000.",
      confidence: "Unavailable",
    });
    apiStatus.textContent = "Backend: unreachable";
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict Disease";
  }
}

function updateDiseaseResult(data) {
  diseaseResult.textContent = data.disease || "Unknown";
  organicSolution.textContent = data.organic_solution || "No organic advice available.";
  chemicalSolution.textContent = data.chemical_solution || "No chemical advice available.";
  confidenceResult.textContent = data.confidence || "Not provided";
}

async function handleChatSubmit(event) {
  event.preventDefault();

  const message = chatInput.value.trim();
  if (!message) {
    return;
  }

  appendMessage(message, "user");
  chatInput.value = "";

  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Chat request failed.");
    }

    appendMessage(data.response, "bot");
    if (voiceReplyEnabled) {
      speakText(data.response);
    }
    apiStatus.textContent = "Backend: connected";
  } catch (error) {
    appendMessage(
      `${error.message} Start the FastAPI backend and try again.`,
      "bot",
    );
    apiStatus.textContent = "Backend: unreachable";
  }
}

function appendMessage(text, sender) {
  const messageElement = document.createElement("div");
  messageElement.className = `message ${sender}`;
  messageElement.textContent = text;
  chatMessages.appendChild(messageElement);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function startVoiceInput() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    appendMessage("Voice input is not supported in this browser.", "bot");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  voiceBtn.disabled = true;
  voiceBtn.textContent = "Listening...";

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    chatInput.value = transcript;
    chatForm.requestSubmit();
  };

  recognition.onerror = () => {
    appendMessage("I could not understand the voice input. Please try again.", "bot");
  };

  recognition.onend = () => {
    voiceBtn.disabled = false;
    voiceBtn.textContent = "Start Voice Input";
  };

  recognition.start();
}

function toggleVoiceReply() {
  voiceReplyEnabled = !voiceReplyEnabled;
  muteBtn.textContent = voiceReplyEnabled ? "Voice Reply On" : "Voice Reply Off";

  if (!voiceReplyEnabled) {
    window.speechSynthesis.cancel();
  }
}

function speakText(text) {
  if (!("speechSynthesis" in window)) {
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  utterance.rate = 1;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

async function checkBackendStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    if (!response.ok) {
      throw new Error("Backend check failed.");
    }
    apiStatus.textContent = "Backend: connected";
  } catch (error) {
    apiStatus.textContent = "Backend: waiting";
  }
}
