let map;
let routeCoords = [];
let userCoords = null;
let markers = [];
let watchId = null;
let activeHotspots = [];
let notifiedHotspots = new Set();
let currentWeather = '';
let isNight = false;
let autocompleteTimeout = null;

// Kozhikode Approximate Bounding Box for validation (Exact Graph Bounds)
const KOZHIKODE_BOUNDS = {
    minLat: 11.125,
    maxLat: 11.805,
    minLon: 75.535,
    maxLon: 76.120
};

// Distance Calculation (Haversine)
function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371e3; // Earth radius in meters
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δφ = (lat2 - lat1) * Math.PI / 180;
    const Δλ = (lon2 - lon1) * Math.PI / 180;

    const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
        Math.cos(φ1) * Math.cos(φ2) *
        Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c; // Distance in meters
}

// Proximity Checker
function checkProximity() {
    if (!userCoords || activeHotspots.length === 0) return;

    activeHotspots.forEach(hotspot => {
        const hotspotId = `${hotspot.lat.toFixed(5)},${hotspot.lon.toFixed(5)}`;
        if (notifiedHotspots.has(hotspotId)) return;

        const distance = calculateDistance(userCoords.lat, userCoords.lon, hotspot.lat, hotspot.lon);

        if (distance <= 300) {
            notifiedHotspots.add(hotspotId);
            const riskLevel = hotspot.prob > 0.85 ? 'High' : (hotspot.prob > 0.55 ? 'Moderate' : 'Low');

            let reasons = [];
            if (currentWeather && currentWeather !== 'clear') reasons.push(`${currentWeather} weather`);
            if (isNight) reasons.push("low visibility at night");
            if (hotspot.features) {
                if (hotspot.features.curvature_score > 1.2) reasons.push("sharp curve ahead");
                if (hotspot.features.is_junction) reasons.push("upcoming junction");
                if (hotspot.features.maxspeed > 50) reasons.push("high-speed zone");
            }

            const reasonStr = reasons.length > 0 ? reasons.join(', ') : "historical accident data";
            const message = `Approaching ${hotspot.name || 'Unknown Road'}.\nRisk: ${riskLevel}\nReason: ${reasonStr}.`;

            triggerSpecificNotification("🚨 Risk Ahead!", message);
        }
    });
}

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

    status.textContent = "Locating (Live)...";

    if (watchId) navigator.geolocation.clearWatch(watchId);

    watchId = navigator.geolocation.watchPosition(
        (pos) => {
            userCoords = { lat: pos.coords.latitude, lon: pos.coords.longitude };
            status.textContent = `✅ Location Acquired (${userCoords.lat.toFixed(4)}, ${userCoords.lon.toFixed(4)})`;

            // Update top button
            const topBtn = document.getElementById('topLocBtn');
            if (topBtn) {
                topBtn.classList.add('granted');
                topBtn.textContent = "GPS Enabled";
            }

            // Clear existing user marker
            if (window.userMarker) map.removeLayer(window.userMarker);
            window.userMarker = L.marker([userCoords.lat, userCoords.lon]).addTo(map).bindPopup("You are here");

            // Check proximity on every move
            checkProximity();
        },
        (err) => {
            status.textContent = "⚠️ Location Error: " + err.message;
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
}

// Show Error Message
function showError(message) {
    const errorMsg = document.getElementById('errorMsg');
    if (message) {
        errorMsg.textContent = message;
        errorMsg.classList.remove('hidden');
    } else {
        errorMsg.classList.add('hidden');
        errorMsg.textContent = '';
    }
}

// Analyze Route
async function analyzeRoute() {
    const origin = document.getElementById('origin').value;
    const destination = document.getElementById('destination').value;
    const btn = document.getElementById('analyzeBtn');
    const status = document.getElementById('status');
    const statusText = document.getElementById('statusText');

    showError(null); // Clear previous errors

    if (!destination) {
        showError("Please enter a destination");
        return;
    }

    // Basic Bounding Box validation via destination name hint (if we had geocoded it, we could check coords, but backend will reject it anyway if it can't find it in Kozhikode graph. Let's add a quick frontend check if the user explicitly typed something far away).
    // The true restriction is handled by the autocomplete viewbox and the backend graph.

    const normOrigin = origin.trim().toLowerCase();
    if ((!normOrigin || normOrigin === 'current location' || normOrigin === 'here' || normOrigin === 'me' || normOrigin === 'default (current location)') && !userCoords) {
        showError("Still acquiring GPS lock... Please ensure you granted location permissions to this site, step outside if indoors, and wait for the ✅ indicator under Location Services before analyzing.");
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
        showError("Error: " + err.message);
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

    // Save for proximity checks
    currentWeather = data.weather || 'clear';
    isNight = data.time_ctx?.is_night || false;

    // Clear old map layers
    markers.forEach(m => map.removeLayer(m));
    markers = [];
    if (window.routeLine) map.removeLayer(window.routeLine);

    // Draw route
    window.routeLine = L.polyline(data.route_coords, { color: 'blue', weight: 5 }).addTo(map);
    map.fitBounds(window.routeLine.getBounds());

    // Display hotspots
    data.top_5_risks.forEach((risk, i) => {
        const riskLevel = risk.prob > 0.85 ? 'High' : (risk.prob > 0.55 ? 'Moderate' : 'Low');
        const riskClass = risk.prob > 0.85 ? 'risk-high' : (risk.prob > 0.55 ? 'risk-mod' : 'risk-low');

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

    // Save hotspots for proximity tracking
    activeHotspots = data.top_5_risks;
    notifiedHotspots.clear(); // Reset notifications for new route

    // Google Maps Link
    gmapsLink.href = data.maps_url;

    // Save for notification
    window.lastRiskText = data.top_5_risks.map((r, i) =>
        `${i + 1}. ${r.name} (${r.prob > 0.85 ? 'High' : (r.prob > 0.55 ? 'Mod' : 'Low')})`
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

async function triggerSpecificNotification(title, textBody) {
    if (!("Notification" in window) || Notification.permission !== 'granted') return;

    const options = {
        body: textBody,
        icon: 'https://unpkg.com/lucide-static@0.263.1/icons/alert-triangle.svg',
        requireInteraction: true,
        vibrate: [200, 100, 200]
    };

    try {
        const reg = await navigator.serviceWorker.ready;
        reg.showNotification(title, options);
    } catch (e) {
        new Notification(title, options);
    }
}

// Autocomplete Logic
function initAutocomplete() {
    // Autocomplete feature has been disabled per request.
}

// Initialize
window.onload = () => {
    initMap();
    getLocation();
    initAutocomplete();

    // Register Service Worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js');
    }

    document.getElementById('analyzeBtn').onclick = analyzeRoute;
    document.getElementById('getLocBtn').onclick = getLocation;
    document.getElementById('notifyBtn').onclick = triggerNotification;

    // Top permission buttons
    document.getElementById('topLocBtn').onclick = getLocation;
    document.getElementById('topNotifyBtn').onclick = async () => {
        const btn = document.getElementById('topNotifyBtn');
        await triggerNotification();
        if (Notification.permission === 'granted') {
            btn.classList.add('granted');
            btn.textContent = "Notifications Enabled";
        }
    };

    // Auto-update GPS button state
    if (userCoords) document.getElementById('topLocBtn').classList.add('granted');
};
