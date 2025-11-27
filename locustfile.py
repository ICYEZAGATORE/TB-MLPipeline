from locust import HttpUser, task, between
import os

class MLApiUser(HttpUser):
    wait_time = between(1, 3)  # seconds between requests
    host = "http://localhost:8000"

    def on_start(self):
        """Called once when a simulated user starts"""
        # Check if sample image exists
        if not os.path.exists("sample.jpg"):
            print("WARNING: sample.jpg not found. Load test may fail.")

    @task(3)  # Weight: 3x more likely than other tasks
    def predict_image(self):
        """Test the /predict endpoint with file upload"""
        try:
            with open("sample.jpg", "rb") as f:
                files = {
                    "file": ("sample.jpg", f, "image/jpeg")
                }
                
                response = self.client.post(
                    "/predict",
                    files=files,
                    name="/predict"  # Groups all predict requests together in stats
                )
                
                # Optional: validate response
                if response.status_code == 200:
                    result = response.json()
                    print(f"✓ Prediction: {result.get('predicted_class', 'N/A')}")
                else:
                    print(f"✗ Failed: {response.status_code}")
                    
        except FileNotFoundError:
            print("ERROR: sample.jpg not found")
            self.environment.runner.quit()

    @task(1)  # Weight: runs less frequently
    def check_health(self):
        """Test the /health endpoint"""
        self.client.get("/health", name="/health")

    @task(1)
    def get_metrics(self):
        """Test the /metrics endpoint"""
        self.client.get("/metrics", name="/metrics")