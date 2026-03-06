from pyngrok import ngrok
import time

try:
    print("Starting Ngrok tunnel to port 5000...")
    # Open a HTTP tunnel on the default port 80
    public_url = ngrok.connect(5000)
    print("=====================================================")
    print(f"YOUR NEW RELIABLE LINK IS -> {public_url.public_url}")
    print("=====================================================")
    print("Leave this terminal open. Do not close it.")
    
    # Keep the tunnel alive
    try:
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()
    except KeyboardInterrupt:
        print("Shutting down tunnel...")
        ngrok.kill()

except Exception as e:
    print(f"Ngrok experienced an error: {e}")
