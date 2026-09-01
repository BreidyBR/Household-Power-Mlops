"""
Monitoreo operativo de la API de inferencia.

Registra las métricas de sistema requeridas para observar el servicio:
- Latency: tiempo promedio de respuesta.
- Throughput: solicitudes procesadas por segundo.
- Error rate: proporción de solicitudes con error.
- Availability: proporción de solicitudes atendidas correctamente.
"""

import time
from threading import Lock


class SystemMonitor:
    """Acumula métricas operativas de la API durante su ejecución."""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.total_requests = 0
        self.error_requests = 0
        self.total_latency_seconds = 0.0
        self._lock = Lock()

    def record_request(self, latency_seconds: float, status_code: int):
        """Registra el resultado y la latencia de una solicitud."""

        with self._lock:
            self.total_requests += 1
            self.total_latency_seconds += latency_seconds

            if status_code >= 500:
                self.error_requests += 1

    def get_metrics(self):
        """Calcula y devuelve las métricas operativas actuales."""

        with self._lock:
            uptime_seconds = time.perf_counter() - self.start_time

            if self.total_requests == 0:
                return {
                    "latency_ms": 0.0,
                    "throughput_rps": 0.0,
                    "error_rate": 0.0,
                    "availability": 1.0,
                    "total_requests": 0,
                }

            average_latency = (
                self.total_latency_seconds / self.total_requests
            ) * 1000

            throughput = (
                self.total_requests / uptime_seconds
                if uptime_seconds > 0
                else 0.0
            )

            error_rate = self.error_requests / self.total_requests
            availability = 1.0 - error_rate

            return {
                "latency_ms": round(average_latency, 3),
                "throughput_rps": round(throughput, 3),
                "error_rate": round(error_rate, 4),
                "availability": round(availability, 4),
                "total_requests": self.total_requests,
            }