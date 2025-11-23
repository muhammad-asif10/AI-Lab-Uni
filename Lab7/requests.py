try:
    import requests
    resp = requests.get('https://api.github.com', timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        print('GitHub API Response (keys):', list(data.keys())[:5])
    else:
        print(f'HTTP Error: {resp.status_code}')
except Exception as e:
    print('Could not import/use requests. Install it with pip and try again.\n', e)
