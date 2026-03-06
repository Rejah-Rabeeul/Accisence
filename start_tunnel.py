import subprocess
import os
import time

# Kill existing stray SSH processes
os.system("taskkill /f /im ssh.exe")

print("Starting tunnel with IPv4 explicit binding...")
p = subprocess.Popen(
    ['ssh', '-R', '80:127.0.0.1:5000', 'noqr@ssh.localhost.run'], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True, 
    encoding='utf-8',
    bufsize=1
)

url = None
with open('tunnel_link.txt', 'w') as f:
    f.write("Connecting...")

for line in iter(p.stdout.readline, ''):
    # Looking for: "5e04cfec2b64b4.lhr.life tunneled with tls termination"
    if '.lhr.life tunneled' in line:
        domain = line.strip().split(' ')[0]
        url = f"https://{domain}"
        with open('tunnel_link.txt', 'w') as f:
            f.write(url)
        print(f"=====================================================")
        print(f"FOUND URL: {url}")
        print(f"=====================================================")
        break

print("Tunnel is up and running in background!")
# Keep alive
while p.poll() is None:
    time.sleep(1)
