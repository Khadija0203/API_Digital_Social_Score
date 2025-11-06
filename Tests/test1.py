import asyncio
import aiohttp
import time
import random
import statistics
from typing import List, Dict
import json

# Configuration
LOCAL_API_URL = "http://localhost:8080"  # API locale
PROD_API_URL = "http://34.68.240.253"   # API production

# Datasets de test variés
TEST_TEXTS = {
    "toxic_high": [
        "I hate you so much, you're stupid",
        "You are an idiot and I want to kill you", 
        "Die in hell, moron",
        "You're worthless trash"
    ],
    "toxic_medium": [
        "You're annoying me",
        "This is stupid",
        "I don't like you",
        "That's really dumb",
        "You make me angry"
    ],
    "non_toxic": [
        "Have a great day!",
        "Thank you for your help",
        "This is really interesting",
        "I love this product",
        "You're doing amazing work",
        "Weather is nice today",
        "I'm learning a lot",
        "This is helpful information"
    ],
    "edge_cases": [
        "",  # Empty text
        "A",  # Very short
        "Lorem ipsum " * 100,  # Very long
        "🚀🎯📊💡🔥",  # Only emojis
        "123 456 789",  # Only numbers
        "https://example.com/test?param=value"  # URLs
    ]
}

class APITester:
    def __init__(self, api_url: str = LOCAL_API_URL):
        self.api_url = api_url
        self.results = []
        
    async def single_request(self, session: aiohttp.ClientSession, text: str, category: str, idx: int):
        """Envoie une requête POST et collecte les métriques"""
        start = time.perf_counter()
        
        try:
            payload = {"text": text}
            async with session.post(
                f"{self.api_url}/predict", 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                response_data = await resp.json()
                elapsed = time.perf_counter() - start
                
                return {
                    "idx": idx,
                    "category": category,
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "status": resp.status,
                    "elapsed": elapsed,
                    "prediction": response_data.get("prediction"),
                    "probability": response_data.get("probability"),
                    "label": response_data.get("label"),
                    "error": None
                }
                
        except Exception as e:
            elapsed = time.perf_counter() - start
            return {
                "idx": idx,
                "category": category, 
                "text": text[:50] + "..." if len(text) > 50 else text,
                "status": None,
                "elapsed": elapsed,
                "prediction": None,
                "probability": None,
                "label": None,
                "error": str(e)
            }

    async def health_check(self, session: aiohttp.ClientSession):
        """Test du health check"""
        try:
            async with session.get(f"{self.api_url}/health") as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def scenario_1_mixed_load(self, concurrency: int = 20, duration: int = 60):
        """Scénario 1: Charge mixte pendant X secondes"""
        print(f"Scénario 1: Charge mixte ({concurrency} req/s pendant {duration}s)")
        
        end_time = time.time() + duration
        request_idx = 0
        
        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                # Sélection aléatoire de textes
                tasks = []
                for _ in range(concurrency):
                    category = random.choice(list(TEST_TEXTS.keys()))
                    text = random.choice(TEST_TEXTS[category])
                    
                    tasks.append(self.single_request(session, text, category, request_idx))
                    request_idx += 1
                
                # Lancer les requêtes et attendre 1 seconde
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                self.results.extend([r for r in batch_results if not isinstance(r, Exception)])
                
                await asyncio.sleep(1)  # 1 requête par seconde
                
        print(f" Scénario 1 terminé: {len(self.results)} requêtes")

    async def scenario_2_spike_test(self, spike_size: int = 100):
        """Scénario 2: Pic de charge soudain"""
        print(f" Scénario 2: Pic de charge ({spike_size} requêtes simultanées)")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(spike_size):
                # Principalement des textes toxiques pour tester le pire cas
                category = "toxic_high" if i % 3 == 0 else "non_toxic"
                text = random.choice(TEST_TEXTS[category])
                tasks.append(self.single_request(session, text, f"spike_{category}", i))
            
            spike_results = await asyncio.gather(*tasks, return_exceptions=True)
            self.results.extend([r for r in spike_results if not isinstance(r, Exception)])
            
        print(f" Scénario 2 terminé: {len([r for r in spike_results if not isinstance(r, Exception)])} requêtes")

    async def scenario_3_edge_cases(self):
        """Scénario 3: Test des cas limites"""
        print(" Scénario 3: Cas limites et erreurs")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, text in enumerate(TEST_TEXTS["edge_cases"] * 3):  # Répéter 3 fois
                tasks.append(self.single_request(session, text, "edge_case", i))
            
            edge_results = await asyncio.gather(*tasks, return_exceptions=True)
            self.results.extend([r for r in edge_results if not isinstance(r, Exception)])
            
        print(f" Scénario 3 terminé: {len([r for r in edge_results if not isinstance(r, Exception)])} requêtes")

    async def scenario_4_health_monitoring(self):
        """Scénario 4: Monitoring santé système"""
        print(" Scénario 4: Tests de santé")
        
        async with aiohttp.ClientSession() as session:
            for i in range(10):
                health = await self.health_check(session)
                print(f"  Health check {i+1}: {health.get('status', 'ERROR')}")
                await asyncio.sleep(2)
                
        print(" Scénario 4 terminé")

    def analyze_results(self):
        """Analyse des résultats pour validation"""
        if not self.results:
            print(" Aucun résultat à analyser")
            return
        
        print("\n ANALYSE DES RÉSULTATS")
        print("=" * 50)
        
        # Statistiques globales
        total = len(self.results)
        successes = [r for r in self.results if r["status"] == 200]
        errors = [r for r in self.results if r["status"] != 200 or r["error"]]
        
        print(f"Total requêtes     : {total}")
        print(f"Succès (200)       : {len(successes)} ({len(successes)/total*100:.1f}%)")
        print(f"Erreurs            : {len(errors)} ({len(errors)/total*100:.1f}%)")
        
        # Analyse par catégorie
        by_category = {}
        for r in self.results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"toxic": 0, "non_toxic": 0, "errors": 0}
                
            if r["error"] or r["status"] != 200:
                by_category[cat]["errors"] += 1
            elif r["label"] == "toxic":
                by_category[cat]["toxic"] += 1
            elif r["label"] == "non_toxic":
                by_category[cat]["non_toxic"] += 1
        
        print("\n Répartition par catégorie:")
        for cat, stats in by_category.items():
            total_cat = sum(stats.values())
            print(f"  {cat:15}: {stats['toxic']:3} toxic, {stats['non_toxic']:3} non-toxic, {stats['errors']:3} erreurs ({total_cat} total)")
        
        # Performance
        latencies = [r["elapsed"] for r in self.results if r["elapsed"] and r["status"] == 200]
        if latencies:
            print(f"\n Performance (requêtes réussies):")
            print(f"  Temps moyen        : {statistics.mean(latencies):.3f}s")
            print(f"  Temps médian       : {statistics.median(latencies):.3f}s")
            print(f"  Temps min/max      : {min(latencies):.3f}s / {max(latencies):.3f}s")
            
            lat_sorted = sorted(latencies)
            if len(lat_sorted) >= 10:
                p95_idx = int(0.95 * len(lat_sorted))
                print(f"  P95                : {lat_sorted[p95_idx-1]:.3f}s")

        print("\n VÉRIFIONS LES DASHBOARDS GRAFANA")
        print("   → http://localhost:3000")

async def main():
    print(" TEST COMPLET DE VISUALISATION - API TOXIC DETECTION")
    print("=" * 60)
    
    # Choisir l'API (locale par défaut)
    api_url = LOCAL_API_URL
    
    # Test de connectivité
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    print(f" Connexion API OK: {api_url}")
                else:
                    print(f" API répond mais statut {resp.status}")
    except:
        print(f" Impossible de contacter {api_url}")
        print("   Vérifiez que votre API fonctionne avec: python app.py")
        return
    
    # Initialiser le testeur
    tester = APITester(api_url)
    
    try:
        # Lancer tous les scénarios
        await tester.scenario_4_health_monitoring()  # D'abord la santé
        await tester.scenario_3_edge_cases()         # Cas limites
        await tester.scenario_1_mixed_load(concurrency=5, duration=30)  # Charge normale
        await tester.scenario_2_spike_test(spike_size=20)  # Pic de charge
        
        # Analyser les résultats
        tester.analyze_results()
        
    except KeyboardInterrupt:
        print("\nTest interrompu par l'utilisateur")
        tester.analyze_results()
    except Exception as e:
        print(f" Erreur durant le test: {e}")

if __name__ == "__main__":
    asyncio.run(main())