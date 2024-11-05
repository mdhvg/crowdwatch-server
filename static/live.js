import { io } from "https://cdn.socket.io/4.8.0/socket.io.esm.min.js";
const socket = io("/");

socket.on("connect", () => {
    console.log("Connected to server");
});

socket.on("disconnect", () => {
    console.log("Disconnected from server");
});

const canvas = document.getElementById("heatmapCanvas");
const ctx = canvas.getContext('2d');
const map = document.getElementById("map");
const container = document.getElementById("container");

socket.on("img", (data) => {
    map.style.background = `url("data:image/png;base64,${data}") no-repeat`
    map.style.backgroundSize = "100% 100%";
})

socket.on("headCoords", (data) => plotPoints(data));

let imgWidth = 640;
let imgHeight = 480;

const plotPoints = (data) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const jsonData = JSON.parse(data);

    const containerWidth = container.offsetWidth;
    const containerHeight = container.offsetHeight;

    for (let i = 0; i < jsonData.length; i++) {
        ctx.fillStyle = 'rgb(255, 0, 0)';
        ctx.beginPath();
        ctx.arc(containerWidth * jsonData[i][0], containerHeight * jsonData[i][1], 5, 0, Math.PI * 2);
        ctx.fill();
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

scaleImage()
setCanvas()