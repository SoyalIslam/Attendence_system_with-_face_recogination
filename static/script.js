let lastLogCount = 0;
let lastSeenName = "";
let allLogs = [];
let scanCheckInterval = null;

async function askBot() {
    const input = document.getElementById('botInput');
    const output = document.getElementById('chatOutput');
    const query = input.value.trim();
    if (!query) return;

    output.innerHTML += `<div class="text-end text-white mb-1">You: ${query}</div>`;
    input.value = '';
    output.scrollTop = output.scrollHeight;

    try {
        const res = await fetch('/ask_bot', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: query})
        });
        const data = await res.json();
        
        output.innerHTML += `<div class="text-info mb-1">AI: ${data.answer}</div>`;
        output.scrollTop = output.scrollHeight;
    } catch (e) {
        output.innerHTML += `<div class="text-danger mb-1">Error connecting to bot.</div>`;
    }
}

async function startScan() {
    const btn = document.getElementById('startBtn');
    const img = document.getElementById('videoFeed');
    const placeholder = document.getElementById('videoPlaceholder');
    
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Starting...';
    
    try {
        const response = await fetch('/start_scan', { method: 'POST' });
        const res = await response.json();
        
        if (res.success) {
            btn.innerHTML = 'Scanning...';
            placeholder.style.display = 'none';
            img.style.display = 'block';
            img.src = videoFeedUrl + "?t=" + new Date().getTime();
            
            if (scanCheckInterval) clearInterval(scanCheckInterval);
            scanCheckInterval = setInterval(checkScanStatus, 1000);
        }
    } catch (e) {
        console.error("Failed to start scan:", e);
        btn.disabled = false;
        btn.innerText = "▶ Start Scan";
        showToast("Failed to start camera", "danger");
    }
}

async function checkScanStatus() {
    try {
        const response = await fetch('/check_scan_status');
        const data = await response.json();
        
        if (!data.active) {
            stopScanUI();
        }
    } catch (e) {
        console.error("Error checking status:", e);
    }
}

function stopScanUI() {
    if (scanCheckInterval) clearInterval(scanCheckInterval);
    
    const btn = document.getElementById('startBtn');
    
    btn.disabled = false;
    btn.innerText = "▶ Start Scan";
    
    showToast("Scan Complete / Stopped");
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('statusToast');
    const toastMsg = document.getElementById('toastMessage');
    
    toast.className = `alert status-toast shadow ${type}`;
    toastMsg.innerText = message;
    toast.style.display = 'block';

    setTimeout(() => {
        toast.style.display = 'none';
    }, 4000);
}

document.getElementById('registerForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const name = document.getElementById('nameInput').value;
    const formData = new FormData();
    formData.append('name', name);

    try {
        const response = await fetch('/register_unknown', { method: 'POST', body: formData });
        const result = await response.json();
        
        const alertBox = document.getElementById('alertBox');
        if (result.success) {
            showToast("User registered successfully", "success");
            alertBox.innerHTML = `<div class="alert alert-success bg-success text-white border-0">User registered successfully!</div>`;
            document.getElementById('nameInput').value = '';
            fetchLogs(); 
        } else {
            alertBox.innerHTML = `<div class="alert alert-danger bg-danger text-white border-0">${result.message}</div>`;
        }
        setTimeout(() => alertBox.innerHTML = '', 3000);
    } catch (error) {
        console.error('Error:', error);
    }
});

async function deleteUser(name) {
    if(!confirm(`Delete user "${name}"?`)) return;

    const formData = new FormData();
    formData.append('name', name);

    try {
        const response = await fetch('/delete_user', { method: 'POST', body: formData });
        const result = await response.json();
        if (result.success) {
            showToast(result.message, "warning");
            fetchLogs();
        } else {
            showToast(result.message, "danger");
        }
    } catch (error) {
        console.error('Error deleting user:', error);
    }
}

function filterLogs() {
    const input = document.getElementById('searchInput').value.toLowerCase();
    const filtered = allLogs.filter(log => 
        log.Name.toLowerCase().includes(input) || 
        log.Date.includes(input)
    );
    renderTable(filtered);
}

function renderTable(logsToRender) {
    const tbody = document.getElementById('logTableBody');
    tbody.innerHTML = '';
    
    logsToRender.forEach(log => {
        const deleteBtn = log.Name !== 'Unknown' 
            ? `<button class="btn btn-sm btn-glass btn-glass-danger py-0" onclick="deleteUser('${log.Name}')">🗑</button>` 
            : '';

        let exitDisplay = log.ExitTime;
        if (log.ExitTime === 'Active') {
            exitDisplay = '<span class="badge bg-success">Active</span>';
        }

        const row = `<tr>
            <td>
                <strong>${log.Name}</strong><br>
                <small class="text-white-50">${log.Date}</small>
            </td>
            <td>${log.EntryTime}</td>
            <td>${exitDisplay}</td>
            <td>${deleteBtn}</td>
        </tr>`;
        tbody.innerHTML += row;
    });
}

async function fetchLogs() {
    try {
        const response = await fetch('/get_logs');
        const logs = await response.json();
        allLogs = logs.slice().reverse();
        
        const searchVal = document.getElementById('searchInput').value;
        if (!searchVal) {
            renderTable(allLogs);
        }

        if (logs.length > lastLogCount && lastLogCount !== 0) {
            const latest = allLogs[0];
            if (latest && latest.Name !== 'Unknown' && latest.Name !== lastSeenName) {
                showToast(`Attendance marked: ${latest.Name}`);
                lastSeenName = latest.Name;
            }
        }
        lastLogCount = logs.length;

    } catch (error) {
        console.error('Error fetching logs:', error);
    }
}

setInterval(fetchLogs, 2000);
fetchLogs();