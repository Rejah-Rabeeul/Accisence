self.addEventListener('install', function (event) {
    self.skipWaiting();
});

self.addEventListener('activate', function (event) {
    event.waitUntil(clients.claim());
});

self.addEventListener('push', function (event) {
    const data = event.data ? event.data.json() : {};
    const title = data.title || "Accisence Alert";
    const options = {
        body: data.body || "New notification",
        icon: data.icon || "https://unpkg.com/lucide-static@0.263.1/icons/alert-triangle.svg",
        requireInteraction: true
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

self.addEventListener('notificationclick', function (event) {
    event.notification.close();
});
