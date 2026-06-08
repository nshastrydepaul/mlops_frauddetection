from locust import HttpUser, between, task


class FraudAPIUser(HttpUser):
    host = "https://fraud-detection-predict-lj4hg275tq-uc.a.run.app"
    wait_time = between(1, 2)

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(2)
    def list_models(self):
        self.client.get("/models")

    @task(5)
    def predict_simple(self):
        self.client.post(
            "/predict/simple?amt=5000&merchant_risk_30_day=28&trans_time_is_night=1&avg_amt_per_customer=50"
        )
