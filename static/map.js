import { io } from "https://cdn.socket.io/4.8.0/socket.io.esm.min.js";
const socket = io("/");

socket.on("connect", () => {
    console.log("Connected to server");
});

socket.on("disconnect", () => {
    console.log("Disconnected from server");
});

socket.on("headCoords", (data) => plotHeatmap(data));

const delay = ms => new Promise(res => setTimeout(res, ms));
const canvas = document.getElementById("heatmapCanvas");
const ctx = canvas.getContext('2d');
const map = document.getElementById("map");
const container = document.getElementById("container");

map.style.background = `url(/map?t=${Math.random()}) no-repeat`;
map.style.backgroundSize = "100% 100%";

let imgWidth = 0;
let imgHeight = 0;
const img = new Image();
img.src = `/map?t=${Math.random()}`;
img.onload = () => {
    imgWidth = img.width;
    imgHeight = img.height;
    scaleImage()
    setCanvas()
    setInterval(() => {
        fetchCoordinates()
    }, 1000)
}

const fetchCoordinates = () => {
    socket.emit("get_headcoords")
}

let xCoords = [];
let yCoords = [];

const plotHeatmap = (data) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const jsonData = JSON.parse(data);
    // console.log(jsonData)
    for (let i = 0; i < jsonData.length; i++) {
        const group = jsonData[i];

        const containerWidth = container.offsetWidth;
        const containerHeight = container.offsetHeight;

        const startRatio = group["startRatio"];
        const diffRatio = group["diffRatio"];

        const relevantWidth = containerWidth * diffRatio[0];
        const relevantHeight = containerHeight * diffRatio[1];

        const offsetX = containerWidth * startRatio[0];
        const offsetY = containerHeight * startRatio[1];

        const xFlip = group["xFlip"];
        const yFlip = group["yFlip"];

        if (group["headCoords"].length > 0) {
            if (xFlip) {
                xCoords = group["headCoords"].map(coord => (1 - coord[0]) * relevantWidth + offsetX);
            }
            else {
                xCoords = group["headCoords"].map(coord => coord[0] * relevantWidth + offsetX);
            }
            if (yFlip) {
                yCoords = group["headCoords"].map(coord => (1 - coord[1]) * relevantHeight + offsetY);
            }
            else {
                yCoords = group["headCoords"].map(coord => coord[1] * relevantHeight + offsetY);
            }
        }

        drawPoints(xCoords, yCoords);

        const heatmap = generateHeatmap(xCoords, yCoords, containerWidth, containerHeight, gridSize);
        drawCircularHeatmap(heatmap, gridSize, ctx);
    }
}

const setCanvas = () => {
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
}

const scaleImage = () => {
    const imageAspect = imgWidth / imgHeight;
    container.style.height = container.offsetWidth / imageAspect + 'px';
}

window.onresize = () => {
    scaleImage()
    setCanvas()
}

const gridSize = 100;

// Function to generate the heatmap data
function generateHeatmap(xCoords, yCoords, width, height, gridSize) {
    const rows = Math.ceil(height / gridSize);
    const cols = Math.ceil(width / gridSize);
    const heatmap = Array.from({ length: rows }, () => Array(cols).fill(0));

    for (let i = 0; i < xCoords.length; i++) {
        const x = xCoords[i];
        const y = yCoords[i];
        const gridX = Math.floor(x / gridSize);
        const gridY = Math.floor(y / gridSize);

        if (gridX >= 0 && gridX < cols && gridY >= 0 && gridY < rows) {
            heatmap[gridY][gridX]++;
        }
    }
    return heatmap;
}

// Function to draw circular heatmap on canvas
async function drawCircularHeatmap(heatmap, gridSize, ctx) {
    for (let row = 0; row < heatmap.length; row++) {
        for (let col = 0; col < heatmap[row].length; col++) {
            const intensity = heatmap[row][col];
            if (intensity > 0) {
                const alpha = Math.min(intensity, 1); // Adjust alpha based on intensity
                const radius = gridSize / 2; // Radius for circular heatmap

                // Create a radial gradient
                const gradient = ctx.createRadialGradient(
                    col * gridSize + radius, // X coordinate for center
                    row * gridSize + radius, // Y coordinate for center
                    0, // Inner radius
                    col * gridSize + radius, // X coordinate for edge
                    row * gridSize + radius, // Y coordinate for edge
                    radius // Outer radius
                );

                // Define gradient color stops
                if (intensity < 3) {
                    gradient.addColorStop(0, `rgba(0, 255, 0, ${alpha})`); // Green
                    gradient.addColorStop(1, `rgba(0, 255, 0, 0)`); // Transparent
                } else if (intensity < 5) {
                    gradient.addColorStop(0, `rgba(255, 255, 0, ${alpha})`); // Yellow
                    gradient.addColorStop(1, `rgba(255, 255, 0, 0)`); // Transparent
                } else {
                    gradient.addColorStop(0, `rgba(255, 0, 0, ${alpha})`); // Red
                    gradient.addColorStop(1, `rgba(255, 0, 0, 0)`); // Transparent
                }

                // Fill the circular area with the gradient
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(col * gridSize + radius, row * gridSize + radius, radius, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }
}

function drawPoints(xCoords, yCoords) {
    for (let i = 0; i < xCoords.length; i++) {
        ctx.fillStyle = 'rgb(255, 0, 0)';
        ctx.beginPath();
        ctx.arc(xCoords[i], yCoords[i], 5, 0, Math.PI * 2);
        ctx.fill();
    }
}