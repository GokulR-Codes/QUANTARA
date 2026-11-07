import subprocess, time, urllib.request, json, sys

p = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '8003'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    cwd=r'd:\Final\quantum_federated_behaviour_ai'
)

for _ in range(30):
    time.sleep(1)
    try:
        urllib.request.urlopen('http://127.0.0.1:8003/api/v1/health', timeout=2)
        print('Server up')
        break
    except Exception:
        pass

base = 'http://127.0.0.1:8003'
for label, path in [('health', '/api/v1/health'), ('ready1', '/api/v1/ready'), ('ready2', '/api/v1/ready')]:
    t0 = time.time()
    try:
        r = urllib.request.urlopen(base + path, timeout=15)
        data = json.loads(r.read())
        elapsed = time.time() - t0
        if 'ready' in label:
            print(label, '->', r.status, f'{elapsed:.2f}s | ready=' + str(data['ready']) + ' quantum=' + str(data['quantum_available']) + ' model=' + str(data['model_available']))
        else:
            print(label, '->', r.status, f'{elapsed:.2f}s |', json.dumps(data)[:100])
    except Exception as e:
        print(label, '-> ERROR:', e)

p.terminate()
p.wait(timeout=5)
