"use strict";
const image = document.getElementById('image');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const blurCanvas = document.createElement('canvas');
const blurCtx = blurCanvas.getContext('2d');
const pixelCanvas = document.createElement('canvas');
const pixelCtx = pixelCanvas.getContext('2d');
const brightCanvas = document.createElement('canvas');
const brightCtx = brightCanvas.getContext('2d');
// const gameCanvas = document.getElementById('game');
// const gameCtx = gameCanvas.getContext('2d');
const markRegions = document.getElementById('markregions');
const fileSelect = document.getElementById("fileSelect");
const fileElem = document.getElementById("fileElem");
const retry = document.getElementById("retry");
const type = document.getElementById("type");
const setting = document.getElementById("setting");

const t = {};
let model;

const log = (msg) => console.log(msg);

const options = {
    modelPath: '../models/default-f16/model.json',
    imagePath: '../samples/example.jpg',
    minScore: 0.38,
    maxResults: 50,
    iouThreshold: 0.5,
    outputNodes: ['output1', 'output2', 'output3'],
    resolution: [801, 1112],
    user: {
        map: {
            blur,
            solid: solidColor,
            bright,
            pixel
        },
        blurStrength: 10,
        brightness: 3.5,
        pixelSize: 3.5,
        fillColor: '#000000',
        censorType: 'pixel'
    }
};

const labels = [
    'exposed anus',  //0
    'exposed armpits',  //1
    'belly',  //2
    'exposed belly',  //3
    'buttocks',  //4
    'exposed buttocks',  //5
    'female face',  //6
    'male face',  //7
    'feet',  //8
    'exposed feet',  //9
    'breast',  //10
    'exposed breast',  //11
    'vagina',  //12
    'exposed vagina',  //13
    'male breast',  //14
    'exposed male breast',  //15
];

const betaSettings = {
    pathetic: {
        person: [1,2,3,4,6,7,8,9,14,15],
        sexy: [5,10,12],
        nude: [0,11,13],
    },
    original: {
        person: [1,2,3,4,6,7,8,9,14,15],
        sexy: [1,3,5,9,10,12,14,15],
        nude: [0,11,13],
    },
    ultimate: {
        person: [7,14,15],
        sexy: [1,2,4,6,8,10],
        nude: [0,3,5,9,11,12,13],
    }
}

let composite = betaSettings.pathetic

async function processPrediction(boxesTensor, scoresTensor, classesTensor, inputTensor) {
    const boxes = await boxesTensor.array();
    const scores = await scoresTensor.data();
    const classes = await classesTensor.data();
    const nmsT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, options.maxResults, options.iouThreshold, options.minScore); // sort & filter results
    const nms = await nmsT.data();
    tf.dispose(nmsT);
    const parts = [];
    for (const i in nms) { // create body parts object
        const id = parseInt(i);
        parts.push({
            score: scores[i],
            id: classes[id],
            class: labels[classes[id]],
            box: [
                Math.trunc(boxes[0][id][0]),
                Math.trunc(boxes[0][id][1]),
                Math.trunc((boxes[0][id][3] - boxes[0][id][1])),
                Math.trunc((boxes[0][id][2] - boxes[0][id][0])),
            ],
        });
    }
    const result = {
        input: { width: inputTensor.shape[2], height: inputTensor.shape[1] },
        person: parts.filter((a) => composite.person.includes(a.id)).length > 0,
        sexy: parts.filter((a) => composite.sexy.includes(a.id)).length > 0,
        nude: parts.filter((a) => composite.nude.includes(a.id)).length > 0,
        parts,
    };
    return result;
}

function blur({left=0, top=0, width=0, height=0}) {
    if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
        return;
    blurCanvas.width = width
    blurCanvas.height = height

    blurCtx.filter = `blur(${options.user.blurStrength}px)`;

    blurCtx.drawImage(canvas, left, top, width, height, 0, 0, width ,height);

    ctx.drawImage(blurCanvas, left, top, width, height);
}

function bright({left=0, top=0, width=0, height=0}) {
    if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
        return;
    brightCanvas.width = width
    brightCanvas.height = height

    brightCtx.filter = `brightness(${options.user.brightness})`;

    brightCtx.drawImage(canvas, left, top, width, height, 0, 0, width ,height);

    ctx.drawImage(brightCanvas, left, top, width, height);
}

function pixel({left=0, top=0, width=0, height=0}) {
    if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
        return;
    pixelCanvas.width = width
    pixelCanvas.height = height

    pixelCtx.drawImage(canvas, left, top, width, height, 0, 0, width ,height);

    let size = options.user.pixelSize / 100,
    w = pixelCanvas.width * size,
    h = pixelCanvas.height * size;

    pixelCtx.drawImage(pixelCanvas, 0, 0, w, h);

    pixelCtx.msImageSmoothingEnabled = false;
    pixelCtx.mozImageSmoothingEnabled = false;
    pixelCtx.webkitImageSmoothingEnabled = false;
    pixelCtx.imageSmoothingEnabled = false;

    pixelCtx.drawImage(pixelCanvas, 0, 0, w, h, 0, 0, pixelCanvas.width, pixelCanvas.height);

    ctx.drawImage(pixelCanvas, left, top, width, height);

    pixelCtx.msImageSmoothingEnabled = true;
    pixelCtx.mozImageSmoothingEnabled = true;
    pixelCtx.webkitImageSmoothingEnabled = true;
    pixelCtx.imageSmoothingEnabled = true;
}

function solidColor({left=0, top=0, width=0, height=0}) {
    if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
        return;
    ctx.fillStyle = options.user.fillColor
    ctx.fillRect(left, top, width, height)
}

function rect({ x=0, y=0, width=0, height=0, radius=8, lineWidth=2, color='white', title='', font='20px "Segoe UI"'}) {
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.lineWidth = 4;
    ctx.fillStyle = color;
    ctx.font = font;
    ctx.fillText(title, x + 4, y - 4);
}

function processParts(res) {
    for (const obj of res.parts) { // draw all detected objects
        if (composite.nude.includes(obj.id))
            options.user.map[options.user.censorType]({ left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3] });
        if (composite.sexy.includes(obj.id))
            options.user.map[options.user.censorType]({ left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3] });
        if (markRegions.checked) {
            rect({ x: obj.box[0], y: obj.box[1], width: obj.box[2], height: obj.box[3], title: `${obj.class}` });
        }
    }
}

async function processLoop() {
    if (canvas.width !== image.width)
        canvas.width = image.width;
    if (canvas.height !== image.height)
        canvas.height = image.height;
    if (canvas.width > 0 && model) {

        t.buffer = await tf.browser.fromPixelsAsync(image);
        t.resize = (options.resolution[0] > 0 && options.resolution[1] > 0 && (options.resolution[0] !== image.width || options.resolution[1] !== image.height)) // do we need to resize
            ? tf.image.resizeNearestNeighbor(t.buffer, [options.resolution[1], options.resolution[0]])
            : t.buffer;
        t.cast = tf.cast(t.resize, 'float32');
        t.batch = tf.expandDims(t.cast, 0);

        [t.boxes, t.scores, t.classes] = await model.executeAsync(t.batch, options.outputNodes);

        const res = await processPrediction(t.boxes, t.scores, t.classes, t.cast);
        await tf.browser.toPixels(t.resize, canvas);
        processParts(res);
    }
}

async function main() {
    if (tf.engine().registryFactory.webgpu && navigator?.gpu)
        await tf.setBackend('webgpu');
    else
        await tf.setBackend('webgl');
    tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true); // doubles the performance
    await tf.ready();

    model = await tf.loadGraphModel(options.modelPath);
    image.src = options.imagePath;
    image.onload = async () => {
        options.resolution[0] = image.width
        options.resolution[1] = image.height

        await processLoop()
        // gameSetup()
    };
}

fileSelect.addEventListener("click", (e) => {
    if (fileElem) {
        fileElem.click();
    }
},false);

function reset() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    pixelCtx.clearRect(0, 0, pixelCanvas.width, pixelCanvas.height);
    blurCtx.clearRect(0, 0, blurCanvas.width, blurCanvas.height);
    brightCtx.clearRect(0, 0, brightCanvas.width, brightCanvas.height);
    // gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
    main()
}

retry.addEventListener('click', () => {
    reset()
})

fileElem.onchange = function(){
    options.imagePath = window.URL.createObjectURL(this.files[0])
    reset()
}

type.onchange = (e) => {
    options.user.censorType = e.target.value
}

setting.onchange = (e) => {
    composite = betaSettings[e.target.value]
}

// function gameSetup() {
//     gameCanvas.width = image.width
//     gameCanvas.height = image.height

//     gameCtx.fillStyle = '#fff'
//     gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height)
// }

main()