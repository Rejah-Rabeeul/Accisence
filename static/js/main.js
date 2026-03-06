let map;
let routeCoords = [];
let userCoords = null;
let markers = [];

// Initialize Map
function initMap() {
    map = L.map('map').setView([11.2588, 75.7804], 13); // Default to Kozhikode
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
}

// Get User Location
function getLocation() {
    const status = document.getElementById('locStatus');
    if (!navigator.geolocation) {
        status.textContent = "Geolocation not supported";
        return;
    }

    status.textContent = "Locating...";
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            userCoords = { lat: pos.coords.latitude, lon: pos.coords.longitude };
            status.textContent = `✅ Location Acquired (${userCoords.lat.toFixed(4)}, ${userCoords.lon.toFixed(4)})`;
            map.setView([userCoords.lat, userCoords.lon], 15);
            L.marker([userCoords.lat, userCoords.lon]).addTo(map).bindPopup("You are here").openPopup();
        },
        (err) => {
            status.textContent = "⚠️ Location Error: " + err.message;
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
}

// Analyze Route
async function analyzeRoute() {
    const origin = document.getElementById('origin').value;
    const destination = document.getElementById('destination').value;
    const btn = document.getElementById('analyzeBtn');
    const status = document.getElementById('status');
    const statusText = document.getElementById('statusText');

    if (!destination) {
        alert("Please enter a destination");
        return;
    }

    const normOrigin = origin.trim().toLowerCase();
    if ((!normOrigin || normOrigin === 'current location' || normOrigin === 'here' || normOrigin === 'me') && !userCoords) {
        alert("Still acquiring GPS lock... Please ensure you granted location permissions to this site, step outside if indoors, and wait for the ✅ indicator under Location Services before analyzing.");
        return;
    }

    btn.disabled = true;
    status.classList.remove('hidden');
    statusText.textContent = "Finding shortest path...";

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                origin: origin,
                destination: destination,
                lat: userCoords ? userCoords.lat : null,
                lon: userCoords ? userCoords.lon : null
            })
        });

        const data = await response.json();
        if (response.status !== 200) throw new Error(data.error || "Failed to analyze");

        displayResults(data);

        // Automatically trigger push notification with the analysis text
        triggerNotification();

    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        status.classList.add('hidden');
    }
}

function displayResults(data) {
    const resultsCard = document.getElementById('resultsCard');
    const hotspotList = document.getElementById('hotspotList');
    const gmapsLink = document.getElementById('gmapsLink');

    resultsCard.classList.remove('hidden');
    hotspotList.innerHTML = '';

    // Overall Risk Section
    const riskLevelSpan = document.getElementById('overallRiskLevel');
    const conditionsSpan = document.getElementById('routeConditions');

    riskLevelSpan.textContent = data.overall_risk || 'Unknown';
    riskLevelSpan.className = 'risk-tag'; // reset
    if (data.overall_risk === 'High') riskLevelSpan.classList.add('risk-high');
    else if (data.overall_risk === 'Medium') riskLevelSpan.classList.add('risk-mod');
    else riskLevelSpan.classList.add('risk-low');

    const isNightText = data.time_ctx?.is_night ? 'Nighttime' : 'Daytime';
    const weatherText = data.weather ? (data.weather.charAt(0).toUpperCase() + data.weather.slice(1)) : 'Unknown Weather';
    conditionsSpan.textContent = `${weatherText} • ${isNightText}`;

    // Clear old map layers
    markers.forEach(m => map.removeLayer(m));
    markers = [];
    if (window.routeLine) map.removeLayer(window.routeLine);

    // Draw route
    window.routeLine = L.polyline(data.route_coords, { color: 'blue', weight: 5 }).addTo(map);
    map.fitBounds(window.routeLine.getBounds());

    // Display hotspots
    data.top_5_risks.forEach((risk, i) => {
        const riskLevel = risk.prob > 0.7 ? 'High' : (risk.prob > 0.4 ? 'Moderate' : 'Low');
        const riskClass = risk.prob > 0.7 ? 'risk-high' : (risk.prob > 0.4 ? 'risk-mod' : 'risk-low');

        // List item
        const item = document.createElement('div');
        item.className = 'hotspot-item';
        item.innerHTML = `
            <span>${i + 1}. ${risk.name || 'Unknown Road'}</span>
            <span class="risk-tag ${riskClass}">${riskLevel}</span>
        `;
        hotspotList.appendChild(item);

        // Map marker
        const marker = L.circleMarker([risk.lat, risk.lon], {
            radius: 8,
            color: 'red',
            fillColor: '#f03',
            fillOpacity: 0.5
        }).addTo(map).bindPopup(`<b>Risk: ${riskLevel}</b><br>${risk.name || 'Road'}`);
        markers.push(marker);
    });

    // Google Maps Link
    gmapsLink.href = data.maps_url;

    // Save for notification
    window.lastRiskText = data.top_5_risks.map((r, i) =>
        `${i + 1}. ${r.name} (${r.prob > 0.7 ? 'High' : 'Mod'})`
    ).join('\n');
}

// Notification System
async function triggerNotification() {
    if (!("Notification" in window)) {
        alert("Notifications not supported");
        return;
    }

    let permission = Notification.permission;
    if (permission === 'default') {
        permission = await Notification.requestPermission();
    }

    if (permission !== 'granted') {
        alert("Permission denied. Results:\n\n" + window.lastRiskText);
        return;
    }

    const title = "🚨 Hotspot Alert";
    const options = {
        body: window.lastRiskText,
        icon: 'https://unpkg.com/lucide-static@0.263.1/icons/alert-triangle.svg',
        requireInteraction: true
    };

    try {
        // Since we are NOT in an iframe now, this should work natively on Android Chrome!
        const reg = await navigator.serviceWorker.ready;
        reg.showNotification(title, options);
    } catch (e) {
        new Notification(title, options);
    }
}

// Initialize
window.onload = () => {
    initMap();
    getLocation();

    // Register Service Worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js');
    }

    document.getElementById('analyzeBtn').onclick = analyzeRoute;
    document.getElementById('getLocBtn').onclick = getLocation;
    document.getElementById('notifyBtn').onclick = triggerNotification;
};
