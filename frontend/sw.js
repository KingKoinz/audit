// ENTROPY Service Worker for Push Notifications

self.addEventListener('install', (event) => {
  console.log('[SW] Service Worker installed');
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  console.log('[SW] Service Worker activated');
  event.waitUntil(clients.claim());
});

self.addEventListener('push', (event) => {
  console.log('[SW] Push received');

  let data = {
    title: 'ENTROPY Alert',
    body: 'A significant finding was detected!',
    icon: '/Entropy.png',
    badge: '/Entropy.png',
    data: { url: '/findings' }
  };

  try {
    if (event.data) {
      data = { ...data, ...event.data.json() };
    }
  } catch (e) {
    console.error('[SW] Failed to parse push data:', e);
  }

  const options = {
    body: data.body,
    icon: data.icon || '/Entropy.png',
    badge: data.badge || '/Entropy.png',
    vibrate: [200, 100, 200, 100, 200],
    tag: 'entropy-finding',
    renotify: true,
    requireInteraction: true,
    actions: [
      { action: 'view', title: 'View Finding' },
      { action: 'dismiss', title: 'Dismiss' }
    ],
    data: data.data
  };

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notification clicked:', event.action);

  event.notification.close();

  if (event.action === 'dismiss') {
    return;
  }

  // Open the findings page
  const url = event.notification.data?.url || '/findings';

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((windowClients) => {
        // Check if there's already a window open
        for (const client of windowClients) {
          if (client.url.includes('/findings') && 'focus' in client) {
            return client.focus();
          }
        }
        // Open new window
        if (clients.openWindow) {
          return clients.openWindow(url);
        }
      })
  );
});
