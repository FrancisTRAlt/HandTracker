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
  // Model loading is handled by ml5 silently
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
  
  // Suppress "Loading..." text by drawing video immediately in draw()
  handPose.detectStart(video, (results) => { hands = results; });
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
  const confirmed = await showPopup("Reset System?", "Warning: All collected data and trained models will be erased.", true);
  if (confirmed) {
    classes = []; classIdx = 0; dataCount = 0;
    isTrained = false; isCollecting = false;
    classification = ""; confidence = 0;
    select('#class-list').html('');
    select('#class-input').value('');
    select('#status-display').html('Waiting for classes...');
    select('#train-progress').hide().html('Processing...');
    select('#result-overlay').hide();
    select('#recording-overlay').hide();
    select('#export-section').addClass('hidden');
    select('#collect-btn').attribute('disabled', '');
    select('#train-btn').attribute('disabled', '');
    initClassifier();
  }
}

async function addNewClass() {
  let input = select('#class-input');
  let val = input.value().trim();
  if (!val || isTrained) return;
  if (val.length > 10) {
    await showPopup("Invalid Name", "Class name must not be longer than 10 characters.", false);
    return;
  }
  if (classes.includes(val)) {
    await showPopup("Duplicate Name", "This class name already exists.", false);
    return;
  }
  classes.push(val);
  let chip = createSpan(val)
    .class("bg-indigo-900/40 text-indigo-200 text-[10px] font-bold px-3 py-1.5 rounded-full border border-indigo-500/30 transition-all")
    .id(`class-chip-${classes.length - 1}`);
  chip.parent('#class-list');
  input.value('');
  updateDataCountUI();
  if (classes.length >= 2) select('#collect-btn').removeAttribute('disabled');
}

function draw() {
  background(10);
  image(video, 0, 0, width, height);
  
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
  stroke(129, 140, 248); strokeWeight(2.5);
  for (let conn of connections) {
    line(hand.keypoints[conn[0]].x, hand.keypoints[conn[0]].y, hand.keypoints[conn[1]].x, hand.keypoints[conn[1]].y);
  }
  fill(255); noStroke();
  for (let kp of hand.keypoints) circle(kp.x, kp.y, 5);
}

function gotClassification(results) {
  if (results && results[0].confidence > 0.05) {
    classification = results[0].label;
    confidence = results[0].confidence;
    select('#result-overlay').show();
    select('#res-label').html(classification);
    select('#res-conf').html(nf(confidence * 100, 0, 0) + "%");
  }
}

function handleDataCollection(inputData) {
  const maxData = int(select('#max-data-input').value());
  
  if (maxData <= 0) {
    isCollecting = false;
    select('#recording-overlay').hide();
    return;
  }

  if (dataCount < maxData) {
    classifier.addData(inputData, { label: classes[classIdx] });
    dataCount++;
  } else {
    let finishedChip = select(`#class-chip-${classIdx}`);
    if (finishedChip) finishedChip.addClass('class-finished');
    isCollecting = false;
    classIdx++;
    dataCount = 0;
    select('#recording-overlay').hide();
    
    if (classIdx >= classes.length) {
      select('#status-display').html("✅ Data Collection Complete");
      select('#train-btn').removeAttribute('disabled');
    }
  }
  updateDataCountUI();
}

function train() {
  const epochs = int(select('#epochs-input').value()) || 50;
  const batchSize = int(select('#batch-input').value()) || 32;
  const learningRate = float(select('#lr-input').value()) || 0.01;

  select('#train-progress').show();
  select('#train-btn').attribute('disabled', '');
  
  classifier.normalizeData();
  
  const options = { epochs, batchSize, learningRate };

  classifier.train(options, (epoch, loss) => {
    select('#train-progress').html(`Training: ${epoch + 1}/${epochs}`);
  }, () => {
    isTrained = true;
    select('#train-progress').html("✨ Training Success!");
    select('#export-section').removeClass('hidden');
  });
}

function startCollection() {
  const maxData = int(select('#max-data-input').value());
  if (maxData <= 0) {
    showPopup("Invalid Input", "Max Samples per class must be at least 1.", false);
    return;
  }

  if (classIdx < classes.length) {
    isCollecting = true;
    select('#recording-overlay').style('display', 'flex');
    select('#target-label').html(classes[classIdx]);
  }
}

function updateDataCountUI() {
  const maxData = select('#max-data-input').value();
  let c = classes[classIdx] || "All Done";
  select('#status-display').html(`Current Class: <b class="text-indigo-400">${c}</b><br>Samples: ${dataCount} / ${maxData}`);
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
        d.push(Math.sqrt(dx * dx + dy * dy), Math.atan2(dy, dx) / PI);
      }
    } else {
      const singleHandLength = 21 * 2 + connections.length * 2;
      for (let i = 0; i < singleHandLength; i++) d.push(0);
    }
  }
  return d;
}

async function downloadZip() {
  const zip = new JSZip();
  const modelFolder = zip.folder("model");
  await classifier.neuralNetwork.model.save({
    save: (artifacts) => {
      modelFolder.file("model.json", JSON.stringify({
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: artifacts.weightsManifest
      }));
      modelFolder.file("model.weights.bin", artifacts.weightData);
      modelFolder.file("model_meta.json", JSON.stringify(classifier.neuralNetworkData.meta));
      return Promise.resolve({ modelArtifactsInfo: artifacts });
    }
  });
  zip.generateAsync({ type: "blob" }).then(content => {
    saveAs(content, "hand_pose_model.zip");
  });
}