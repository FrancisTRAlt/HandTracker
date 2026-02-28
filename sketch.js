let classifier, handPose, video;
let hands = [];
let dataCount = 0;
let isTrained = false;
let classification = "";
let confidence = 0;
let connections;
let classes = [];
let classIdx = 0;
let isCollecting = false;

function showPopup(title, body, showCancel = true) {
    return new Promise((resolve) => {
        select('#modal-title').html(title);
        select('#modal-body').html(body);
        select('#custom-modal').addClass('modal-active');
        if (showCancel) select('#modal-cancel').show();
        else select('#modal-cancel').hide();
        select('#modal-confirm').mousePressed(() => {
            select('#custom-modal').removeClass('modal-active');
            resolve(true);
        });
        select('#modal-cancel').mousePressed(() => {
            select('#custom-modal').removeClass('modal-active');
            resolve(false);
        });
    });
}

function preload() {
    handPose = ml5.handPose({ flipped: true, maxHands: 2 });
}

function setup() {
    const container = select('#canvas-container');
    const canvas = createCanvas(640, 480);
    canvas.parent(container);

    video = createCapture(VIDEO, { flipped: true });
    video.size(640, 480);
    video.hide();

    initUI();
    initClassifier();

    handPose.detectStart(video, (results) => {
        hands = results;
        const loader = document.getElementById('loader');
        if (loader) {
            loader.style.opacity = '0';
            setTimeout(() => loader.remove(), 500);
        }
    });
    connections = handPose.getConnections();
}

function initClassifier() {
    ml5.setBackend("webgl");
    classifier = ml5.neuralNetwork({ task: "classification", debug: false });
}

function initUI() {
    select('#add-class-btn').mousePressed(addNewClass);
    select('#collect-btn').mousePressed(startCollection);
    select('#train-btn').mousePressed(train);
    select('#save-btn').mousePressed(downloadZip);
    select('#reset-btn').mousePressed(resetApp);
}

async function resetApp() {
    const confirmed = await showPopup("Reset System?", "Warning: This will clear all progress and reload the page.", true);
    if (confirmed) location.reload();
}

async function addNewClass() {
    let input = select('#class-input');
    let val = input.value().trim();
    if (!val || isTrained) return;
    if (val.length > 15) {
        await showPopup("Invalid Name", "Keep class names under 15 characters.", false);
        return;
    }
    if (classes.includes(val)) {
        await showPopup("Duplicate", "Class name already exists.", false);
        return;
    }
    classes.push(val);
    let chip = createSpan(val)
        .class("bg-indigo-900/40 text-indigo-200 text-xs font-bold px-3 py-1.5 rounded-xl border border-indigo-500/30")
        .id(`class-chip-${classes.length - 1}`);
    chip.parent('#class-list');
    input.value('');
    updateDataCountUI();
    if (classes.length >= 2) select('#collect-btn').removeAttribute('disabled');
}

function draw() {
    clear();
    if (video) image(video, 0, 0, width, height);

    if (hands.length > 0) {
        for (let hand of hands) drawHandSkeleton(hand);
        const inputData = flattenHandData(hands);
        if (isTrained) classifier.classify(inputData, gotClassification);
        if (isCollecting && classIdx < classes.length) handleDataCollection(inputData);
    } else {
        select('#result-overlay').hide();
    }
}

function drawHandSkeleton(hand) {
    stroke(129, 140, 248); strokeWeight(3);
    for (let conn of connections) {
        line(hand.keypoints[conn[0]].x, hand.keypoints[conn[0]].y, hand.keypoints[conn[1]].x, hand.keypoints[conn[1]].y);
    }
    fill(255); noStroke();
    for (let kp of hand.keypoints) circle(kp.x, kp.y, 6);
}

function gotClassification(results) {
    if (results && results[0].confidence > 0.1) {
        classification = results[0].label;
        confidence = results[0].confidence;
        select('#result-overlay').show();
        select('#res-label').html(classification);
        let perc = Math.floor(confidence * 100);
        select('#res-conf').html(perc + "%");
        select('#conf-bar').style('width', perc + '%');
    }
}

function handleDataCollection(inputData) {
    const maxData = int(select('#max-data-input').value());
    if (dataCount < maxData) {
        classifier.addData(inputData, { label: classes[classIdx] });
        dataCount++;
    } else {
        isCollecting = false;
        classIdx++;
        dataCount = 0;
        select('#recording-overlay').hide();
        if (classIdx >= classes.length) {
            select('#status-display').html("✅ Dataset Ready");
            select('#train-btn').removeAttribute('disabled');
        }
    }
    updateDataCountUI();
}

function train() {
    const epochs = int(select('#epochs-input').value()) || 50;
    const batchSize = int(select('#batch-input').value()) || 32;
    const learningRate = float(select('#lr-input').value()) || 0.01;

    select('#train-btn').html("Training...").attribute('disabled', '');
    classifier.normalizeData();

    classifier.train({ epochs, batchSize, learningRate }, (epoch, loss) => {
        select('#train-btn').html(`Epoch: ${epoch + 1}`);
    }, () => {
        isTrained = true;
        select('#train-btn').html("Model Ready").addClass('bg-indigo-600 text-white');
        select('#export-section').removeClass('hidden');
    });
}

function startCollection() {
    if (classIdx < classes.length) {
        isCollecting = true;
        select('#recording-overlay').style('display', 'flex');
        select('#target-label').html("Capturing: " + classes[classIdx]);
    }
}

function updateDataCountUI() {
    const maxData = select('#max-data-input').value();
    let c = classes[classIdx] || "Dataset Ready";
    select('#status-display').html(`${c}: ${dataCount}/${maxData}`);
}

function flattenHandData(allHands) {
    let d = [];
    for (let h = 0; h < 2; h++) {
        if (allHands[h]) {
            const kp = allHands[h].keypoints;
            const x = kp.map(k => k.x), y = kp.map(k => k.y);
            const minX = Math.min(...x), minY = Math.min(...y);
            const w = (Math.max(...x) - minX) || 1, h_s = (Math.max(...y) - minY) || 1;
            for (let k of kp) d.push((k.x - minX) / w, (k.y - minY) / h_s);
            for (let [a, b] of connections) {
                let dx = (kp[b].x - kp[a].x) / w, dy = (kp[b].y - kp[a].y) / h_s;
                d.push(Math.sqrt(dx * dx + dy * dy), Math.atan2(dy, dx) / Math.PI);
            }
        } else {
            const length = 21 * 2 + connections.length * 2;
            for (let i = 0; i < length; i++) d.push(0);
        }
    }
    return d;
}

async function downloadZip() {
    const zip = new JSZip();
    const modelFolder = zip.folder("model");
    await classifier.neuralNetwork.model.save({
        save: (artifacts) => {
            modelFolder.file("model.json", JSON.stringify(artifacts));
            modelFolder.file("model.weights.bin", artifacts.weightData);
            modelFolder.file("model_meta.json", JSON.stringify(classifier.neuralNetworkData.meta));
            return Promise.resolve({ modelArtifactsInfo: artifacts });
        }
    });
    zip.generateAsync({ type: "blob" }).then(content => { saveAs(content, "handtracker_model.zip"); });
}