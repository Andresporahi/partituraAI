// ============================================================
// Partituras AI – Frontend Application
// Compatible con Flask local y Vercel serverless
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

    // BPM controls
    const bpmToggleAuto = document.getElementById("bpmToggleAuto");
    const bpmToggleManual = document.getElementById("bpmToggleManual");
    const bpmInputWrapper = document.getElementById("bpmInputWrapper");
    const bpmInput = document.getElementById("bpmInput");
    const bpmMinus = document.getElementById("bpmMinus");
    const bpmPlus = document.getElementById("bpmPlus");

    let selectedFile = null;
    let bpmMode = "auto"; // "auto" or "manual"

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

    function base64ToBlob(b64Data, contentType) {
        const byteChars = atob(b64Data);
        const byteArrays = [];
        for (let offset = 0; offset < byteChars.length; offset += 512) {
            const slice = byteChars.slice(offset, offset + 512);
            const byteNumbers = new Array(slice.length);
            for (let i = 0; i < slice.length; i++) {
                byteNumbers[i] = slice.charCodeAt(i);
            }
            byteArrays.push(new Uint8Array(byteNumbers));
        }
        return new Blob(byteArrays, { type: contentType });
    }

    function createDownloadUrl(b64Data, contentType) {
        const blob = base64ToBlob(b64Data, contentType);
        return URL.createObjectURL(blob);
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

    // ---- BPM Controls ----
    bpmToggleAuto.classList.add("active");

    bpmToggleAuto.addEventListener("click", () => {
        bpmMode = "auto";
        bpmToggleAuto.classList.add("active");
        bpmToggleManual.classList.remove("active");
        bpmInputWrapper.hidden = true;
    });

    bpmToggleManual.addEventListener("click", () => {
        bpmMode = "manual";
        bpmToggleManual.classList.add("active");
        bpmToggleAuto.classList.remove("active");
        bpmInputWrapper.hidden = false;
        bpmInput.focus();
    });

    bpmMinus.addEventListener("click", () => {
        let val = parseInt(bpmInput.value) || 120;
        bpmInput.value = Math.max(30, val - 1);
    });

    bpmPlus.addEventListener("click", () => {
        let val = parseInt(bpmInput.value) || 120;
        bpmInput.value = Math.min(300, val + 1);
    });

    // ---- Conversion ----
    async function startConversion() {
        if (!selectedFile) return;

        showSection(processingSection);

        const steps = [
            { text: "Subiendo archivo de audio...", progress: 10 },
            { text: "Analizando pitch con CREPE (IA)...", progress: 25 },
            { text: "Detectando energía y silencios reales...", progress: 40 },
            { text: "Extrayendo notas estables de la voz...", progress: 55 },
            { text: "Corrigiendo octavas y fusionando...", progress: 70 },
            { text: "Cuantizando ritmo y generando MIDI...", progress: 82 },
            { text: "Creando partitura musical...", progress: 92 },
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
            if (bpmMode === "manual") {
                const bpmVal = parseInt(bpmInput.value);
                if (bpmVal >= 30 && bpmVal <= 300) {
                    formData.append("bpm", bpmVal);
                }
            }

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

        downloadGrid.innerHTML = "";

        const downloadIcon = `<svg viewBox="0 0 24 24" fill="none"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

        // Modo serverless (base64 en files_data) o modo Flask (URLs en files)
        const isServerless = !!data.files_data;

        if (isServerless) {
            // --- Vercel serverless: archivos como base64 ---
            if (data.files_data.midi) {
                const url = createDownloadUrl(data.files_data.midi.data, data.files_data.midi.mime);
                downloadGrid.innerHTML += `
                    <a href="${url}" class="download-card" download="${data.files_data.midi.filename}">
                        ${downloadIcon}
                        <div class="dl-info">
                            <span class="dl-label">Archivo MIDI</span>
                            <span class="dl-format">.mid</span>
                        </div>
                    </a>`;
            }

            if (data.files_data.musicxml) {
                const url = createDownloadUrl(data.files_data.musicxml.data, data.files_data.musicxml.mime);
                downloadGrid.innerHTML += `
                    <a href="${url}" class="download-card" download="${data.files_data.musicxml.filename}">
                        ${downloadIcon}
                        <div class="dl-info">
                            <span class="dl-label">MusicXML</span>
                            <span class="dl-format">.musicxml</span>
                        </div>
                    </a>`;

                // Render OSMD from base64
                renderOSMDFromBase64(data.files_data.musicxml.data);
            }

            if (data.files_data.png) {
                const url = createDownloadUrl(data.files_data.png.data, data.files_data.png.mime);
                downloadGrid.innerHTML += `
                    <a href="${url}" class="download-card" download="${data.files_data.png.filename}">
                        ${downloadIcon}
                        <div class="dl-info">
                            <span class="dl-label">Partitura PNG</span>
                            <span class="dl-format">.png</span>
                        </div>
                    </a>`;

                scoreImage.src = url;
                scorePreview.hidden = false;
            } else {
                scorePreview.hidden = true;
            }

        } else {
            // --- Flask local: URLs de descarga ---
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

                renderOSMD(data.files.musicxml.replace("/api/download/", "/api/preview/"));
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

                scoreImage.src = data.files.png.replace("/api/download/", "/api/preview/");
                scorePreview.hidden = false;
            } else {
                scorePreview.hidden = true;
            }
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

    async function renderOSMDFromBase64(b64Data) {
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

            const xmlText = atob(b64Data);
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
