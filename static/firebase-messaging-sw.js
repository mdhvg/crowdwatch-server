// Give the service worker access to Firebase Messaging.
// Note that you can only use Firebase Messaging here. Other Firebase libraries
// are not available in the service worker.
// Replace 10.13.2 with latest version of the Firebase JS SDK.
importScripts('https://www.gstatic.com/firebasejs/10.13.2/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.13.2/firebase-messaging-compat.js');

// Initialize the Firebase app in the service worker by passing in
// your app's Firebase config object.
// https://firebase.google.com/docs/web/setup#config-object
firebase.initializeApp({
    apiKey: "AIzaSyAMrPKwdA4It_zGyN8vzP9pXweCZSAEbmA",
    authDomain: "crowd-notif.firebaseapp.com",
    projectId: "crowd-notif",
    storageBucket: "crowd-notif.firebasestorage.app",
    messagingSenderId: "296961942599",
    appId: "1:296961942599:web:7807c7a40ddf4d5db7fd62"
});

// Retrieve an instance of Firebase Messaging so that it can handle background
// messages.
const messaging = firebase.messaging();

messaging.onBackgroundMessage((payload) => {
    console.log(
        '[firebase-messaging-sw.js] Received background message ',
        payload
    );

    self.registration.showNotification(payload.notification.title, {
        body: payload.notification.body,
        icon: payload.notification.image,
    })

    self.registration.showNotification(notificationTitle, notificationOptions);
});