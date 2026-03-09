import subprocess
import time
import os
import re

def start_ngrok():
    print("Starting Ngrok tunnel to port 5000 using direct CLI...")
    
    # Kill existing ngrok processes
    os.system("taskkill /f /im ngrok.exe >nul 2>&1")
    
    # We'll use the region specified in ngrok.yml (India)
    # Adding --log=stdout to capture the URL
    cmd = ['ngrok', 'http', '5000', '--log=stdout']
    
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    url = None
    print("Connecting to Ngrok...")
    
    try:
        # We'll wait up to 30 seconds for a link
        start_time = time.time()
        while time.time() - start_time < 30:
            line = p.stdout.readline()
            if not line:
                break
            print(line.strip())
            
            # Look for the URL in the log output (handles .app and .dev)
            match = re.search(r'url=(https://[a-zA-Z0-9.-]+\.ngrok-free\.(app|dev))', line)
            if not match:
                # Fallback for older ngrok format
                match = re.search(r'url=(https://[a-zA-Z0-9.-]+\.ngrok\.io)', line)
            if match:
                url = match.group(1)
                print("\n" + "="*50)
                print(f"YOUR NEW RELIABLE LINK IS -> {url}")
                print("=====================================================")
                with open('tunnel_link.txt', 'w') as f:
                    f.write(url)
                break
            
            if "heartbeat timeout" in line or "context deadline exceeded" in line:
                print("\nWARNING: Network connectivity issue detected.")
        
        if not url:
            print("\nNgrok is taking a long time or failing. Check the logs above.")
            
        # Keep alive
        while p.poll() is None:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down Ngrok...")
        p.terminate()

if __name__ == "__main__":
    start_ngrok()
