# load_test_50.py
import asyncio
import aiohttp
import time
import statistics
from typing import List
 
API_URL = "http://34.68.240.253/predict"   # endpoint public fourni   http://34.68.240.253/
CONCURRENCY = 50                         # nombre de requêtes simultanées
TIMEOUT = 30                             # timeout par requête en secondes
PAYLOAD = {"text": "I am fed up of this. I will kill everybody"}       # JSON envoyé à l'API
 
async def single_request(session: aiohttp.ClientSession, idx: int):
    """Envoie une requête POST JSON et retourne (status, elapsed_seconds)."""
    start = time.perf_counter()
    try:
        async with session.post(API_URL, json=PAYLOAD, timeout=TIMEOUT) as resp:
            # Essaie de lire le JSON pour forcer la réception complète
            try:
                await resp.json()
            except Exception:
                # si le serveur ne renvoie pas JSON valide, on ignore l'exception de parsing
                pass
            elapsed = time.perf_counter() - start
            return {"idx": idx, "status": resp.status, "elapsed": elapsed, "error": None}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"idx": idx, "status": None, "elapsed": elapsed, "error": str(e)}
 
async def run_load_test(concurrency: int = CONCURRENCY) -> List[dict]:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    connector = aiohttp.TCPConnector(limit=0)  # pas de limite supplémentaire côté client
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [single_request(session, i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
    return results
 
def summarize(results: List[dict]):
    latencies = [r["elapsed"] for r in results if r["elapsed"] is not None]
    successes = [r for r in results if r["status"] is not None and 200 <= r["status"] < 300]
    errors = [r for r in results if r["error"] is not None or (r["status"] is None) or (r["status"] >= 400)]
    statuses = {}
    for r in results:
        key = r["status"]
        statuses[key] = statuses.get(key, 0) + 1
 
    print("=== Résumé du test ===")
    print(f"Requêtes totales   : {len(results)}")
    print(f"Succès (2xx)       : {len(successes)}")
    print(f"Erreurs/échecs     : {len(errors)}")
    print("Codes HTTP reçus   :", statuses)
    if latencies:
        print(f"Temps min (s)      : {min(latencies):.4f}")
        print(f"Temps moyen (s)    : {statistics.mean(latencies):.4f}")
        print(f"Temps médian (s)   : {statistics.median(latencies):.4f}")
        print(f"Temps max (s)      : {max(latencies):.4f}")
        print(f"Écart-type (s)     : {statistics.pstdev(latencies):.4f}")
        # percentiles simples
        lat_sorted = sorted(latencies)
        p90 = lat_sorted[int(0.9 * len(lat_sorted)) - 1] if len(lat_sorted) >= 10 else None
        p95 = lat_sorted[int(0.95 * len(lat_sorted)) - 1] if len(lat_sorted) >= 20 else None
        if p90:
            print(f"P90 (approx) (s)   : {p90:.4f}")
        if p95:
            print(f"P95 (approx) (s)   : {p95:.4f}")
    print("======================")
 
def main():
    print(f"Lancement du test: {CONCURRENCY} requêtes simultanées vers {API_URL}")
    start_total = time.perf_counter()
    results = asyncio.run(run_load_test(CONCURRENCY))
    total_elapsed = time.perf_counter() - start_total
    print(f"Durée totale du test: {total_elapsed:.3f}s")
    summarize(results)
 
if __name__ == "__main__":
    main()
 
 