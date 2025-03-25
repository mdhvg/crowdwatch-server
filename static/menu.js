fetch('/clear', {
    method: 'GET'
}).then(r => r.json()).then(json => console.log(json))

import { initializeApp } from 'https://www.gstatic.com/firebasejs/11.5.0/firebase-app.js'
import { getMessaging, getToken, onMessage } from 'https://www.gstatic.com/firebasejs/11.5.0/firebase-messaging.js'

const tiles = document.querySelectorAll('.tile.video');

tiles.forEach(tile => {
    tile.addEventListener('click', async () => {
        const image = tile.querySelector('img');
        let mapName = image.getAttribute("data-id")
        await fetch('/setMap', {
            method: 'POST',
            body: JSON.stringify({ tile: mapName }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
    })
});

// const liveTile = document.querySelector('.tile.live');

// liveTile.addEventListener('click', async () => {
//     await fetch('/setMap', {
//         method: 'POST',
//         body: JSON.stringify({ tile: 'Live' }),
//         headers: {
//             'Content-Type': 'application/json'
//         }
//     })
// })

const firebaseConfig = {
    apiKey: "AIzaSyAMrPKwdA4It_zGyN8vzP9pXweCZSAEbmA",
    authDomain: "crowd-notif.firebaseapp.com",
    projectId: "crowd-notif",
    storageBucket: "crowd-notif.firebasestorage.app",
    messagingSenderId: "296961942599",
    appId: "1:296961942599:web:7807c7a40ddf4d5db7fd62"
};

initializeApp(firebaseConfig);

const messaging = getMessaging()

const genToken = async () => {
    let token = localStorage.getItem('token');
    if (!token) {
        const permission = await Notification.requestPermission()
        if (permission === "granted") {
            token = await getToken(
                messaging, { vapidKey: "BNy0067s1gfKpc5sIczJLXUuWNFf0BzlQybJ6y2euNF7oJWCu56VZz-LxwpFKCSr7nC93RPLkPE8WmRBbxjhGuM" }
            )
            console.log("Token:", token);
            localStorage.setItem('token', token)
            const response = await fetch("/save_key", {
                method: "POST",
                body: JSON.stringify({ token }),
                mode: "same-origin",
            });
            console.log("/save_key", response.status);
        } else {
            throw new Error("Permission denied")
        }
    }
}

genToken();

onMessage(messaging, (payload) => {
    if (!document.hidden) {
        new Notification(payload.notification.title, {
            body: payload.notification.body,
            icon: payload.notification.image,
            silent: false,
            requireInteraction: false
        })
    }
})