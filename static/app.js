// ============================================================
// Partituras AI – Frontend Application
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzoneContent = document.getElementById("dropzoneContent");
    const dropzoneFile = document.getElementById("dropzoneFile");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const btnRemove = document.getElementById("btnRemove");
    const audioPlayer = document.getElementById("audioPlayer");
    const audioElement = document.getElementById("audioElement");
    const btnConvert = document.getElementById("btnConvert");

    const uploadSection = document.getElementById("uploadSection");
    const processingSection = document.getElementById("processingSection");
    const processingStep = document.getElementById("processingStep");
    const progressFill = document.getElementById("progressFill");
    const resultsSection = document.getElementById("resultsSection");
    const errorSection = document.getElementById("errorSection");
    const errorMessage = document.getElementById("errorMessage");

    const statDuration = document.getElementById("statDuration");
    const statNotes = document.getElementById("statNotes");
    const statKey = document.getElementById("statKey");
    const statBpm = document.getElementById("statBpm");
    const scorePreview = document.getElementById("scorePreview");
    const scoreImage = document.getElementById("scoreImage");
    const osmdContainer = document.getElementById("osmdContainer");
    const osmdRender = document.getElementById("osmdRender");
    const downloadGrid = document.getElementById("downloadGrid");

    const btnNewConversion = document.getElementById("btnNewConversion");
    const btnRetry = document.getElementById("btnRetry");

    let selectedFile = null;

    // ---- Helpers ----
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function showSection(section) {
        [uploadSection, processingSection, resultsSection, errorSection].forEach(s => {
            s.hidden = true;
        });
        section.hidden = false;
        section.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ---- File Selection ----
    function handleFile(file) {
        if (!file) return;

        const allowedTypes = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
                              "audio/flac", "audio/x-flac", "audio/m4a", "audio/mp4",
                              "audio/webm", "audio/x-wav", "audio/wave"];
        const allowedExts = ["wav", "mp3", "ogg", "flac", "m4a", "webm"];
        const ext = file.name.split(".").pop().toLowerCase();

        if (!allowedTypes.includes(file.type) && !allowedExts.includes(ext)) {
            alert("Formato de audio no soportado. Usa WAV, MP3, OGG, FLAC, M4A o WebM.");
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);

        dropzoneContent.hidden = true;
        dropzoneFile.hidden = false;
        dropzone.style.borderStyle = "solid";
        dropzone.style.borderColor = "var(--primary)";
        dropzone.style.background = "var(--primary-light)";

        // Audio preview
        const url = URL.createObjectURL(file);
        audioElement.src = url;
        audioPlayer.hidden = false;

        btnConvert.disabled = false;
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = "";
        dropzoneContent.hidden = false;
        dropzoneFile.hidden = true;
        dropzone.style.borderStyle = "";
        dropzone.style.borderColor = "";
        dropzone.style.background = "";
        audioPlayer.hidden = true;
        audioElement.src = "";
        btnConvert.disabled = true;
    }

    // Dropzone events
    dropzone.addEventListener("click", (e) => {
        if (e.target.closest("#btnRemove")) return;
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    btnRemove.addEventListener("click", (e) => {
        e.stopPropagation();
        clearFile();
    });

    // ---- Conversion ----
    async function startConversion() {
        if (!selectedFile) return;

        showSection(processingSection);

        // Simulate progress steps
        const steps = [
            { text: "Subiendo archivo de audio...", progress: 10 },
            { text: "Detectando tempo del audio...", progress: 20 },
            { text: "Analizando frecuencias vocales (pYIN)...", progress: 35 },
            { text: "Suavizando pitch y corrigiendo octavas...", progress: 50 },
            { text: "Detectando tonalidad musical...", progress: 65 },
            { text: "Cuantizando ritmo y generando MIDI...", progress: 78 },
            { text: "Creando partitura musical...", progress: 90 },
        ];

        let stepIndex = 0;
        const stepInterval = setInterval(() => {
            if (stepIndex < steps.length) {
                processingStep.textContent = steps[stepIndex].text;
                progressFill.style.width = steps[stepIndex].progress + "%";
                stepIndex++;
            }
        }, 1500);

        try {
            const formData = new FormData();
            formData.append("audio", selectedFile);

            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            clearInterval(stepInterval);
            progressFill.style.width = "100%";
            processingStep.textContent = "Finalizando...";

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Error desconocido del servidor.");
            }

            await new Promise(r => setTimeout(r, 500));
            showResults(data);

        } catch (err) {
            clearInterval(stepInterval);
            showError(err.message);
        }
    }

    btnConvert.addEventListener("click", startConversion);

    // ---- Results ----
    function showResults(data) {
        statDuration.textContent = data.duration_seconds;
        statNotes.textContent = data.notes_detected;

        if (data.analysis) {
            statKey.textContent = data.analysis.key_name || data.analysis.key || "--";
            statBpm.textContent = data.analysis.bpm || "--";
        }

        // Clear previous downloads
        downloadGrid.innerHTML = "";

        const downloadIcon = `<svg viewBox="0 0 24 24" fill="none"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

        if (data.files.midi) {
            downloadGrid.innerHTML += `
                <a href="${data.files.midi}" class="download-card" download>
                    ${downloadIcon}
                    <div class="dl-info">
                        <span class="dl-label">Archivo MIDI</span>
                        <span class="dl-format">.mid</span>
                    </div>
                </a>`;
        }

        if (data.files.musicxml) {
            downloadGrid.innerHTML += `
                <a href="${data.files.musicxml}" class="download-card" download>
                    ${downloadIcon}
                    <div class="dl-info">
                        <span class="dl-label">MusicXML</span>
                        <span class="dl-format">.musicxml</span>
                    </div>
                </a>`;
        }

        if (data.files.png) {
            downloadGrid.innerHTML += `
                <a href="${data.files.png}" class="download-card" download>
                    ${downloadIcon}
                    <div class="dl-info">
                        <span class="dl-label">Partitura PNG</span>
                        <span class="dl-format">.png</span>
                    </div>
                </a>`;

            // Show image preview
            scoreImage.src = data.files.png.replace("/api/download/", "/api/preview/");
            scorePreview.hidden = false;
        } else {
            scorePreview.hidden = true;
        }

        // Render MusicXML with OpenSheetMusicDisplay
        if (data.files.musicxml) {
            renderOSMD(data.files.musicxml.replace("/api/download/", "/api/preview/"));
        }

        showSection(resultsSection);
    }

    async function renderOSMD(musicxmlUrl) {
        try {
            if (typeof opensheetmusicdisplay === "undefined") {
                osmdContainer.hidden = true;
                return;
            }

            osmdRender.innerHTML = "";
            osmdContainer.hidden = false;

            const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(osmdRender, {
                autoResize: true,
                backend: "svg",
                drawTitle: true,
                drawComposer: true,
            });

            const response = await fetch(musicxmlUrl);
            const xmlText = await response.text();
            await osmd.load(xmlText);
            osmd.render();
        } catch (err) {
            console.warn("No se pudo renderizar con OSMD:", err);
            osmdContainer.hidden = true;
        }
    }

    // ---- Error ----
    function showError(message) {
        errorMessage.textContent = message;
        showSection(errorSection);
    }

    // ---- Navigation ----
    btnNewConversion.addEventListener("click", () => {
        clearFile();
        scorePreview.hidden = true;
        osmdContainer.hidden = true;
        osmdRender.innerHTML = "";
        showSection(uploadSection);
    });

    btnRetry.addEventListener("click", () => {
        showSection(uploadSection);
    });
});
