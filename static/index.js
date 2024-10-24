import { io } from "https://cdn.socket.io/4.8.0/socket.io.esm.min.js";
const socket = io("http://localhost:8000");

socket.on("connect", () => {
    console.log("Connected to server");
    socket.emit("get_headcoords");
});

socket.on("disconnect", () => {
    console.log("Disconnected from server");
});

socket.on("headcoords", (data) => {
    console.log(data);
})

const delay = ms => new Promise(res => setTimeout(res, ms));
const canvas = document.getElementById("heatmapCanvas");
const ctx = canvas.getContext('2d');
const map = document.getElementById("map");
const container = document.getElementById("container");
let coordsArray = []
let startRatio = [0, 0]
let diffRatio = [0, 0]

let imgWidth = 0;
let imgHeight = 0;
const img = new Image();
img.src = "http://localhost:8000/map.png"
img.onload = () => {
    imgWidth = img.width;
    imgHeight = img.height;
    scaleImage()
    setCanvas()
    drawPoints()
    setInterval(() => {
        getCoords().then(() => { simulateContinuousDataInput() })
    }, 1000)
    // getCoords().then(() => { simulateContinuousDataInput() })
}

const setImage = () => {
    const containerWidth = container.offsetWidth;
    const containerHeight = container.offsetHeight;

    const containerAspectRatio = containerWidth / containerHeight;

    if (containerWidth > containerHeight) {
        map.style.backgroundSize = `100% ${100 * (containerAspectRatio)}%`;
    } else {
        map.style.backgroundSize = `${100 * (1 / containerAspectRatio)}% 100%`;
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

// setImage()

window.onresize = () => {
    scaleImage()
    setCanvas()
    drawPoints()
    // setImage()
}

const gridSize = 5;

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
async function drawCircularHeatmap(heatmap, gridSize, ctx, xCoords = xCoords, yCoords = yCoords) {
    cur = 0;


    for (let row = 0; row < heatmap.length; row++) {
        for (let col = 0; col < heatmap[row].length; col++) {
            const intensity = heatmap[row][col];
            if (intensity > 0) {
                const alpha = 1; // Adjust alpha based on intensity
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
                    // gradient.addColorStop(1, `rgba(0, 255, 0, 0)`); // Transparent
                } else if (intensity < 5) {
                    gradient.addColorStop(0, `rgba(255, 255, 0, ${alpha})`); // Yellow
                    //gradient.addColorStop(1, `rgba(255, 255, 0, 0)`); // Transparent
                } else {
                    gradient.addColorStop(0, `rgba(255, 0, 0, ${alpha})`); // Red
                    // gradient.addColorStop(1, `rgba(255, 0, 0, 0)`); // Transparent
                }

                // Fill the circular area with the gradient
                ctx.fillStyle = 'rgb(255, 0, 0)';
                ctx.beginPath();
                console.log(xCoords[cur], yCoords[cur])
                ctx.arc(xCoords[cur], yCoords[cur], radius, 0, Math.PI * 2);
                ctx.fill();
                cur++;
            }
        }
    }
    console.log(cur)
}

function drawPoints() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < coordsArray.length; i++) {
        console.log(coordsArray[i])
        ctx.fillStyle = 'rgb(255, 0, 0)';
        ctx.beginPath();
        ctx.arc(coordsArray[i][0], coordsArray[i][1], 5, 0, Math.PI * 2);
        ctx.fill();
    }
}

async function simulateContinuousDataInput() {
    // Update the points with new coordinates
    const containerWidth = container.offsetWidth;
    const containerHeight = container.offsetHeight;

    const relevantWidth = containerWidth * diffRatio[0];
    const relevantHeight = containerHeight * diffRatio[1];

    coordsArray = coordsArray.map(coord => {
        const x = coord[0] * relevantWidth + containerWidth * startRatio[0];
        const y = coord[1] * relevantHeight + containerHeight * startRatio[1];
        return [x, y];
    });

    drawPoints();

    // Generate and draw the heatmap
    // const heatmap = generateHeatmap(xCoords, yCoords, canvas.width, canvas.height, gridSize);
    // drawCircularHeatmap(heatmap, gridSize, ctx, xCoords, yCoords);

    // Request the next animation frame to simulate continuous updates
    // await delay(500);
    // requestAnimationFrame(getCoords);
}

async function getCoords() {
    const response = await fetch(window.location.href + 'headcoords')
    if (response.ok) {
        const data = await response.json()
        startRatio = data["startRatio"]
        diffRatio = data["diffRatio"]
        coordsArray = data["headcoords"]
    }
}